# GTR TensorRT deployment

This folder exports all 24 GTR models (6 tasks, 4 sizes) to ONNX, builds FP16
TensorRT engines, and measures their latency. GLA is implemented by the custom
plugin in `trt_plugin/`.


## Reference environment

| Component | Version |
|---|---|
| Board | NVIDIA DRIVE AGX Thor |
| GPU | Blackwell (`sm_110`) |
| OS | DRIVE OS 7.0.5.0 / Ubuntu 24.04.3 LTS |
| CUDA | 13.0 |
| TensorRT | 10.14.2.2 |
| Precision | FP16 |

Use this exact platform to compare latency with the paper. Other NVIDIA GPUs
can run the same workflow, but their latency is not directly comparable.

## Install

Install the GTR Python dependencies from the repository root, then add the ONNX
packages:

```bash
python -m pip install -r requirements.txt
python -m pip install onnx==1.19.1 onnxsim==0.4.36
```

Install the Python bindings shipped with the same TensorRT installation. Do not
mix TensorRT headers, libraries, `trtexec`, and Python bindings from different
versions.

Set the local paths and build the plugin for Thor:

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export TRT_ROOT=/path/to/TensorRT-10.14.2.2
export TRTEXEC="$TRT_ROOT/bin/trtexec"
export PATH="$TRT_ROOT/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$TRT_ROOT/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

cd tensorrt_plugin
SM=110 bash trt_plugin/build.sh
```

`TRT_ROOT` must contain `include/`, `lib/`, and `bin/trtexec`. TensorRT engines
must be built on the target device; do not copy an engine built on another GPU.

## Test one model

This runs GTR-S detection from export to verification:

```bash
mkdir -p out/{onnx,engines,results}

python export_onnx.py \
  -c ../configs/det/coco_finetune/gtr_s.yml \
  -o out/onnx/gtr_s_fp16.onnx \
  --dtype fp16

python bench_trtexec.py \
  --onnx out/onnx/gtr_s_fp16.onnx \
  --engine out/engines/gtr_s_fp16.engine \
  --json out/results/gtr_s_fp16.json \
  --opt-level 5 --extra --maxAuxStreams=4

python verify_trt.py \
  -c ../configs/det/coco_finetune/gtr_s.yml \
  --engine out/engines/gtr_s_fp16.engine \
  --json out/results/gtr_s_fp16.verify.json
```

The last command compares TensorRT with the FP32 PyTorch reference. For a query
head, use `sorted_cos` rather than raw `cos`, because query order may change.

## Reproduce all results

Batch size 1, including numerical verification:

```bash
python run_all.py --all --dtypes fp16 --verify --fallback \
  --opt-level 5 --trt-args=--maxAuxStreams=4 --out out/bs1

python make_report.py \
  --results out/bs1/results --out out/bs1/RESULTS.md
```

Batch size 8:

```bash
python run_all.py --all --dtypes fp16 \
  --export-args="--batch 8" \
  --opt-level 5 --trt-args=--maxAuxStreams=4 --out out/bs8

python make_report.py \
  --results out/bs8/results --out out/bs8/RESULTS.md
```


## Expected latency

| Task | Input | Batch 1 | Batch 8 |
|---|---:|---:|---:|
| Detection | 640² | 2.282 / 2.721 / 3.527 / 4.080 | 14.017 / 19.126 / 27.817 / 32.701 |
| Instance segmentation | 640² | 3.474 / 4.363 / 5.130 / 5.703 | 22.085 / 31.529 / 40.203 / 45.142 |
| Oriented detection | 1024² | 4.310 / 5.491 / 7.602 / 8.769 | 39.619 / 54.312 / 74.545 / 85.553 |
| Pose estimation | 640² | 2.459 / 3.082 / 3.909 / 4.455 | 16.227 / 23.108 / 31.877 / 37.228 |
| Semantic segmentation | 1024² | 3.579 / 4.655 / 6.792 / 7.916 | 36.953 / 50.860 / 70.900 / 81.234 |
| Depth estimation | 640² | 2.675 / 3.071 / 3.872 / 4.389 | 19.212 / 23.953 / 32.588 / 37.328 |



