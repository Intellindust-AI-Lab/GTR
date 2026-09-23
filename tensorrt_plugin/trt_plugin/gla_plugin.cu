// TensorRT IPluginV3 plugin wrapping the GTR hand-written chunk-GLA CUDA operator.


#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "gla_chunk.cuh"

using namespace nvinfer1;

namespace {

constexpr char const* kPluginName = "GatedLinearAttention";
constexpr char const* kPluginVersion = "1";
constexpr size_t kAlign = 256;

inline size_t alignUp(size_t x) { return (x + kAlign - 1) / kAlign * kAlign; }

// Byte offsets of the five kernel workspace regions (same regions the torch
// extension carves out of one fp32 buffer, here padded to 256B boundaries).
struct WsLayout {
    size_t eG, h, S, eL, fl, flBytes, total;
};

WsLayout wsLayout(int B, int T, int H) {
    const int NT = (T + gla_chunk::kBT - 1) / gla_chunk::kBT;
    const size_t K = gla_chunk::kK, V = gla_chunk::kV;
    WsLayout L{};
    size_t off = 0;
    L.eG = off; off += alignUp(size_t(B) * T * H * K * sizeof(float));
    L.h  = off; off += alignUp(size_t(B) * (NT + 1) * H * K * V * sizeof(float));
    L.S  = off; off += alignUp(size_t(B) * NT * H * K * V * sizeof(float));
    L.eL = off; off += alignUp(size_t(B) * NT * H * K * sizeof(float));
    L.flBytes = (size_t(B) * H * NT + 16) * sizeof(float);
    L.fl = off; off += alignUp(L.flBytes);
    L.total = off + kAlign;   // slack so the base pointer can be aligned up
    return L;
}

struct GLAParams {
    float scale{1.0f};
    float gkNormalizer{0.0f};
    float rmsEps{0.0f};
};

}  // namespace

class GLAPlugin : public IPluginV3,
                  public IPluginV3OneCore,
                  public IPluginV3OneBuild,
                  public IPluginV3OneRuntime {
public:
    explicit GLAPlugin(GLAParams p) : mP(p) { initFieldsToSerialize(); }
    GLAPlugin(GLAPlugin const& o) : mP(o.mP) { initFieldsToSerialize(); }

    void initFieldsToSerialize() {
        mFields.clear();
        mFields.emplace_back(PluginField{"scale", &mP.scale, PluginFieldType::kFLOAT32, 1});
        mFields.emplace_back(PluginField{"gk_normalizer", &mP.gkNormalizer, PluginFieldType::kFLOAT32, 1});
        mFields.emplace_back(PluginField{"rms_eps", &mP.rmsEps, PluginFieldType::kFLOAT32, 1});
        mFC.nbFields = static_cast<int32_t>(mFields.size());
        mFC.fields = mFields.data();
    }

    // ---- IPluginV3 ----
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    }
    IPluginV3* clone() noexcept override { return new GLAPlugin(*this); }

    // ---- IPluginV3OneCore ----
    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }

    // ---- IPluginV3OneBuild ----
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                            int32_t) noexcept override {
        return 0;
    }

    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t,
                                   int32_t) noexcept override {
        return inOut[pos].desc.format == TensorFormat::kLINEAR && inOut[pos].desc.type == DataType::kHALF;
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t, DataType const* inputTypes,
                               int32_t) const noexcept override {
        outputTypes[0] = inputTypes[2];
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t,
                            DimsExprs* outputs, int32_t, IExprBuilder&) noexcept override {
        if (nbInputs < 3) return -1;
        outputs[0] = inputs[2];   // same as v: [B, T, H, V]
        return 0;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t, DynamicPluginTensorDesc const*,
                            int32_t) const noexcept override {
        Dims const& d = inputs[0].max;   // static shapes in our engines, but max is always valid
        if (d.nbDims != 4) return 0;
        return wsLayout(static_cast<int>(d.d[0]), static_cast<int>(d.d[1]), static_cast<int>(d.d[2])).total;
    }

    // ---- IPluginV3OneRuntime ----
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
        return 0;
    }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFC; }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        Dims const& qd = inputDesc[0].dims;
        Dims const& vd = inputDesc[2].dims;
        if (qd.nbDims != 4 || vd.nbDims != 4 || workspace == nullptr) return -1;
        const int B = static_cast<int>(qd.d[0]);
        const int T = static_cast<int>(qd.d[1]);
        const int H = static_cast<int>(qd.d[2]);
        const int K = static_cast<int>(qd.d[3]);
        const int V = static_cast<int>(vd.d[3]);
        if (K != gla_chunk::kK || V != gla_chunk::kV || inputDesc[0].type != DataType::kHALF) return -1;
        if (!gla_chunk::chunk_path_supported_on_current_device()) return -1;

        const WsLayout L = wsLayout(B, T, H);
        char* base = reinterpret_cast<char*>(alignUp(reinterpret_cast<size_t>(workspace)));
        float* eG = reinterpret_cast<float*>(base + L.eG);
        float* h  = reinterpret_cast<float*>(base + L.h);
        float* S  = reinterpret_cast<float*>(base + L.S);
        float* eL = reinterpret_cast<float*>(base + L.eL);
        float* fl = reinterpret_cast<float*>(base + L.fl);

        // The fused scan/fwd kernel synchronises producer and consumer blocks through
        // per-chunk flags that must start at zero. TensorRT shares the workspace between
        // layers, so it is re-zeroed on every call (a few hundred bytes, graph-capturable).
        if (cudaMemsetAsync(fl, 0, L.flBytes, stream) != cudaSuccess) return -1;

        const bool fused = mP.rmsEps > 0.0f;
        gla_chunk::launch_chunk_gla(
            static_cast<__half const*>(inputs[0]),   // q
            static_cast<__half const*>(inputs[1]),   // k
            static_cast<__half const*>(inputs[2]),   // v
            static_cast<__half const*>(inputs[3]),   // gk
            static_cast<__half*>(outputs[0]),        // o (or gated y)
            eG, h, B, T, H, mP.scale, stream, S, eL, mP.gkNormalizer,
            fused ? static_cast<__half const*>(inputs[4]) : nullptr,   // g
            fused ? static_cast<__half const*>(inputs[5]) : nullptr,   // rms_w
            fused ? mP.rmsEps : 1e-6f,
            fl);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
    }

private:
    GLAParams mP;
    std::vector<PluginField> mFields;
    PluginFieldCollection mFC{};
};

class GLAPluginCreator : public IPluginCreatorV3One {
public:
    GLAPluginCreator() {
        mAttrs.clear();
        mAttrs.emplace_back(PluginField{"scale", nullptr, PluginFieldType::kFLOAT32, 1});
        mAttrs.emplace_back(PluginField{"gk_normalizer", nullptr, PluginFieldType::kFLOAT32, 1});
        mAttrs.emplace_back(PluginField{"rms_eps", nullptr, PluginFieldType::kFLOAT32, 1});
        mFC.nbFields = static_cast<int32_t>(mAttrs.size());
        mFC.fields = mAttrs.data();
    }
    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }

    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        GLAParams p;
        if (fc != nullptr && fc->fields != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                PluginField const& f = fc->fields[i];
                if (f.data == nullptr) continue;
                float val = 0.0f;
                if (f.type == PluginFieldType::kFLOAT32) val = *static_cast<float const*>(f.data);
                else if (f.type == PluginFieldType::kFLOAT64) val = static_cast<float>(*static_cast<double const*>(f.data));
                else continue;
                if (std::strcmp(f.name, "scale") == 0) p.scale = val;
                else if (std::strcmp(f.name, "gk_normalizer") == 0) p.gkNormalizer = val;
                else if (std::strcmp(f.name, "rms_eps") == 0) p.rmsEps = val;
            }
        }
        return new GLAPlugin(p);
    }

private:
    PluginFieldCollection mFC{};
    std::vector<PluginField> mAttrs;
};

REGISTER_TENSORRT_PLUGIN(GLAPluginCreator);
