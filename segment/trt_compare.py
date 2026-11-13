"""Python reference results for the C++ TensorRT YOLOv5-seg model (CYoloV5SegModel in edu_cpp/TensorRT).

Runs the PyTorch model with the same input as the C++ code (square letterbox at the ONNX size, image centered,
gray 114 padding) and the same post-processing, and writes one <stem>_PyOutYoloV5Seg.json per image in the format
the C++ compareOutputs() reads:
    boxes_rect [x, y, w, h] (original image, float), boxes_class_id, boxes_confidence,
    masks_rect [x, y, w, h] (original image), masks_rle [first value 0/1, run lengths...] row-major inside masks_rect

Settings: trt_compare section of Config/main_config.yaml (confidence, NMS and mask threshold from pixel). Run:
    python segment/trt_compare.py
then the C++ program in normal mode with --python-output <output folder>/json.
"""
from datetime import datetime
import glob
import json
import os
import sys

# Never let YOLOv5 pip-install missing packages into the shared environment. Must be set before utils.general loads.
os.environ.setdefault('YOLOv5_AUTOINSTALL', 'False')

import cv2
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.common import DetectMultiBackend  # noqa: E402
from segment.eval_iou import to_original  # noqa: E402
from segment.model_report import make_mask_decoder, prepare_logger  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.general import non_max_suppression, scale_boxes  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
SUFFIX, TAG = '_PyOutYoloV5Seg', 'yolov5seg_results_python'


def rle(mask):
    """[first value 0/1, run lengths...] of a boolean mask, row-major (same as the C++ rle())."""
    flat = mask.ravel().astype(np.uint8)
    if not flat.size:
        return []
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    return [int(flat[0])] + np.diff(bounds).astype(int).tolist()


@torch.no_grad()
def main(config):
    tc, pc = config.trt_compare, config.pixel
    device = select_device(config.model.device)
    model = DetectMultiBackend(tc.weights, device=device)
    decode_masks = make_mask_decoder(config)
    size = int(tc.imgsz)
    out_dir = os.path.join(config.dataset.test_output, 'json')
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in glob.glob(os.path.join(tc.images, '*')) if f.lower().endswith(IMG_EXT))
    if tc.num_images:
        files = files[:int(tc.num_images)]
    logger.info(f'{len(files)} images, square input {size}, conf {pc.conf_thres}, NMS {pc.iou_nms}, '
                f'max_det {pc.max_det}')
    for i, f in enumerate(files, 1):
        im0 = cv2.imread(f)
        h, w = im0.shape[:2]
        im, _, pad = letterbox(im0, size, auto=False)  # same input as the C++ LETTERBOX_CENTER_NORM
        x = torch.from_numpy(im.transpose(2, 0, 1)[::-1].copy()).to(device).float()[None] / 255
        pred, proto = model(x)[:2]
        det = non_max_suppression(pred, pc.conf_thres, pc.iou_nms, max_det=pc.max_det, nm=proto.shape[1])[0]
        masks = decode_masks(proto[0], det[:, 6:].float(), det[:, :4].float(), x.shape[2:]) if len(det) \
            else det[:0, 0:0]
        mboxes, crops = to_original(masks, pad, h, w)
        boxes = scale_boxes(x.shape[2:], det[:, :4].clone(), (h, w)).cpu().numpy() if len(det) else np.zeros((0, 4))
        res = {'boxes_rect': [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
               'boxes_class_id': det[:, 5].int().cpu().tolist(),
               'boxes_confidence': det[:, 4].float().cpu().tolist(),
               'masks_rect': [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in mboxes],
               'masks_rle': [rle(c) for c in crops]}
        stem = os.path.splitext(os.path.basename(f))[0]
        with open(os.path.join(out_dir, stem + SUFFIX + '.json'), 'w') as fh:
            json.dump({TAG: res}, fh)
        if i % 50 == 0 or i == len(files):
            logger.info(f'{i}/{len(files)} images')
    logger.info(f'Python results in {out_dir}')


if __name__ == '__main__':
    # Load configuration
    config = OmegaConf.load('Config/main_config.yaml')
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(config.environment.cuda_launch_blocking)
    if not config.get('model_name'):
        config.model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(config.trt_compare.weights))))

    # Generate timestamp and run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = os.path.join('bin', f'Run_{timestamp}_{config.model_name}_trt_compare')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Update config paths
    config.dataset.save_path = checkpoints_dir
    config.dataset.test_output = checkpoints_dir

    # Save the config used for this run
    OmegaConf.save(config, os.path.join(checkpoints_dir, "config_used.yaml"))

    prepare_logger(checkpoints_dir, 'trt_compare.log')
    logger.info(f'TensorRT comparison output directory: {checkpoints_dir}')
    main(config)
