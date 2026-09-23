"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

Image / video inference for every GTR task. The task is read from the config
(`task: detection | segmentation | obb | pose`), which decides how the deploy-mode
postprocessor output is unpacked and how it is drawn:

    detection    labels, boxes (N, 4) xyxy,            scores
    segmentation labels, boxes (N, 4) xyxy,            scores, masks (N, h, w) logits
    obb          labels, boxes (N, 5) cx cy w h theta, scores
    pose         scores, labels,                       keypoints (N, K, 2)
"""

import os
import sys

import cv2  # Added for video processing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

# COCO 17-keypoint skeleton, 0-indexed.
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6),
]

# Anything not listed here is handed to the video path. TIFF matters for OBB: aerial
# imagery is routinely distributed as .tif.
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']

PALETTE = [
    (255, 56, 56), (255, 157, 51), (255, 212, 59), (151, 217, 60), (61, 219, 134),
    (26, 195, 189), (52, 147, 255), (99, 100, 255), (172, 87, 255), (255, 87, 190),
]


def class_names(cfg):
    """Class-id -> display name for the config's dataset, or None to fall back to ids."""
    task = cfg.yaml_cfg['task']
    if task == 'obb':
        from engine.data.dataset.dota_dataset import DOTA_CLASSES
        return list(DOTA_CLASSES)
    if task == 'pose':
        # CocoPoseDetection keeps the raw COCO category_id, and person is 1; slot 0 is
        # the unused class of the 2-way head.
        return ['', 'person']
    if cfg.yaml_cfg.get('remap_mscoco_category'):
        from engine.data.dataset.coco_dataset import mscoco_category2name, mscoco_label2category
        return [mscoco_category2name[mscoco_label2category[i]] for i in range(len(mscoco_label2category))]
    return None


def _label_text(names, label, score):
    idx = int(label)
    name = names[idx] if names is not None and idx < len(names) and names[idx] else str(idx)
    return f'{name} {score:.2f}'


def _color(label):
    return PALETTE[int(label) % len(PALETTE)]


def draw_boxes(im, labels, boxes, scores, names):
    """Axis-aligned boxes, xyxy in original-image pixels."""
    d = ImageDraw.Draw(im)
    for label, box, score in zip(labels, boxes, scores):
        c = _color(label)
        d.rectangle(list(box), outline=c, width=2)
        d.text((box[0], max(0, box[1] - 11)), _label_text(names, label, score), fill=c)


def draw_rboxes(im, labels, boxes, scores, names):
    """Oriented boxes, (cx, cy, w, h, theta-radians) in original-image pixels."""
    from engine.data.dataset.dota_eval import rbox_to_poly_np

    d = ImageDraw.Draw(im)
    polys = rbox_to_poly_np(np.asarray(boxes, dtype=np.float64))
    for label, poly, score in zip(labels, polys, scores):
        c = _color(label)
        d.polygon([tuple(p) for p in poly], outline=c)
        top = poly[poly[:, 1].argmin()]  # anchor the caption at the topmost vertex
        d.text((top[0], max(0, top[1] - 11)), _label_text(names, label, score), fill=c)


def draw_pose(im, labels, keypoints, scores, names, radius=3):
    """Keypoints (K, 2) in original-image pixels, plus the COCO skeleton."""
    d = ImageDraw.Draw(im)
    for label, kpts, score in zip(labels, keypoints, scores):
        c = _color(label)
        for a, b in COCO_SKELETON:
            if a < len(kpts) and b < len(kpts):
                d.line([tuple(kpts[a]), tuple(kpts[b])], fill=c, width=2)
        for x, y in kpts:
            d.ellipse([x - radius, y - radius, x + radius, y + radius], fill=c)
        x0, y0 = kpts[:, 0].min(), kpts[:, 1].min()
        d.text((x0, max(0, y0 - 11)), _label_text(names, label, score), fill=c)


def overlay_masks(im, masks, labels, alpha=0.5):
    """Per-query mask logits at head resolution -> binary overlay at image resolution."""
    m = torch.as_tensor(masks)[None].float()
    m = F.interpolate(m, size=(im.size[1], im.size[0]), mode='bilinear', align_corners=False)
    m = (m.sigmoid()[0] > 0.5).numpy()
    canvas = np.array(im).astype(np.float32)
    for label, mask in zip(labels, m):
        if mask.any():
            canvas[mask] = (1 - alpha) * canvas[mask] + alpha * np.array(_color(label), dtype=np.float32)
    im.paste(Image.fromarray(canvas.astype(np.uint8)))


def render(im, task, output, names, thrh):
    """Unpack the deploy-mode postprocessor output for `task` and draw it onto `im`."""
    if task == 'pose':
        scores, labels, keypoints = output
        scores, labels, keypoints = scores[0], labels[0], keypoints[0]
        keep = scores > thrh
        draw_pose(im, labels[keep].cpu(), keypoints[keep].cpu().numpy(), scores[keep].cpu(), names)
        return int(keep.sum())

    labels, boxes, scores = output[0], output[1], output[2]
    labels, boxes, scores = labels[0], boxes[0], scores[0]
    keep = scores > thrh
    labels, boxes, scores = labels[keep].cpu(), boxes[keep].cpu(), scores[keep].cpu()
    if task == 'segmentation' and len(output) > 3:
        overlay_masks(im, output[3][0][keep].cpu(), labels)
    if task == 'obb':
        draw_rboxes(im, labels, boxes.numpy(), scores, names)
    else:
        draw_boxes(im, labels, boxes, scores, names)
    return int(keep.sum())


def build_transforms(size, normalize, resize=True):
    ops = [T.Resize(size)] if resize else []
    ops.append(T.ToTensor())
    if normalize:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(ops)


def preprocess(im_pil, size, transforms, task, device):
    """Image -> (batched tensor, orig_target_sizes) for the postprocessor.

    det / seg / pose keep the plain anisotropic resize their val pipelines use: axis-aligned
    boxes and points stay axis-aligned boxes and points under non-uniform scaling, so the
    postprocessor's per-axis (W, H) rescale inverts it exactly.

    OBB cannot: a rotated rectangle squashed by different x and y factors becomes a
    parallelogram, and OBBPostProcessor can only emit (cx, cy, w, h, theta). Rescaling
    cx/cy/w/h per axis while leaving theta alone therefore misplaces the corners -- ~119 px
    on a 45-degree box in a 2000x1000 image. So OBB is letterboxed instead: one isotropic
    scale, padded bottom-right. Because the pad sits at the far edge, the inverse is a
    single divide by that scale, which is exactly what the postprocessor does when handed
    a target size of (S / scale); theta stays valid. Square inputs (every DOTA patch) are
    unaffected either way.
    """
    w, h = im_pil.size
    if task != 'obb':
        return transforms(im_pil).unsqueeze(0).to(device), torch.tensor([[w, h]]).to(device)

    out_h, out_w = size
    scale = min(out_w / w, out_h / h)
    canvas = Image.new('RGB', (out_w, out_h))
    canvas.paste(im_pil.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                               Image.BILINEAR), (0, 0))
    orig_size = torch.tensor([[out_w / scale, out_h / scale]])
    return transforms(canvas).unsqueeze(0).to(device), orig_size.to(device)


@torch.no_grad()
def process_image(model, device, file_path, out_path, size, normalize, task, names, thrh):
    im_pil = Image.open(file_path).convert('RGB')

    transforms = build_transforms(size, normalize, resize=(task != 'obb'))
    im_data, orig_size = preprocess(im_pil, size, transforms, task, device)
    output = model(im_data, orig_size)

    n = render(im_pil, task, output, names, thrh)
    im_pil.save(out_path)
    print(f'{n} instance(s) above {thrh}; saved {out_path}')


@torch.no_grad()
def process_video(model, device, file_path, out_path, size, normalize, task, names, thrh):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))

    transforms = build_transforms(size, normalize, resize=(task != 'obb'))

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        im_data, orig_size = preprocess(frame_pil, size, transforms, task, device)
        output = model(im_data, orig_size)

        # Draw detections on the frame
        render(frame_pil, task, output, names, thrh)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Video processing complete. Result saved as '{out_path}'.")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    # The whole model is restored from --resume below, so the backbone's "weights not
    # loaded, running with RANDOM parameters" banner would be untrue here. Same
    # suppression train.py applies when resuming.
    # Both backbone flavours: plain ViTAdapter and the Spatial SwiGLU variant.
    for _bb in ('ViTAdapter', 'ViTAdapterSpatialSwiGLU'):
        if _bb in cfg.yaml_cfg:
            cfg.yaml_cfg[_bb]['skip_weights_warning'] = True

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    task = cfg.yaml_cfg['task']
    names = class_names(cfg)
    # Every GTR backbone is an ImageNet-normalized ViT and every val pipeline in
    # configs/ ends with the same Normalize; skipping it here would feed raw [0, 1]
    # pixels to a model trained on normalized input.
    normalize = any(k in cfg.yaml_cfg for k in ('ViTAdapter', 'ViTAdapterSpatialSwiGLU'))

    # Check if the input file is an image or a video
    file_path = args.input
    is_image = os.path.splitext(file_path)[-1].lower() in IMAGE_EXTS
    out_path = args.output or ('torch_results.jpg' if is_image else 'torch_results.mp4')
    if is_image:
        # Process as image
        process_image(model, device, file_path, out_path, img_size, normalize, task, names, args.thrh)
        print("Image processing complete.")
    else:
        # Process as video
        process_video(model, device, file_path, out_path, img_size, normalize, task, names, args.thrh)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, default=None)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-t', '--thrh', type=float, default=0.45,
                        help='score threshold; pose scores run lower than detection, try ~0.3')
    args = parser.parse_args()
    main(args)
