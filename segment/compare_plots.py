"""Side-by-side images of the validation set at full original resolution: labels (GT) on the left, predictions on the
right, with a header showing the per-image counts and pixel metrics.

Settings come from Config/main_config.yaml: model, dataset, pixel (confidence, NMS, mask_mode, mask_thres) and plots
(alpha, gt_color, show_conf), plus the compare_plots section. Run from the repo root:
    python segment/compare_plots.py

Outputs in bin/Run_<timestamp>_<model_name>_compare/:
    config_used.yaml, compare.log
    images/<index>_<image name>.jpg   GT | prediction, full resolution
    compare.csv                       one row per image, worst first (most FP + FN, then lowest pixel IoU)
"""
from datetime import datetime
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
from segment.eval_iou import instance_stats, list_images, load_gt, mean, pixel_stats, to_original  # noqa: E402
from segment.model_report import make_mask_decoder, prepare_logger, write_csv  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.dataloaders import img2label_paths  # noqa: E402
from utils.general import check_dataset, check_img_size, non_max_suppression  # noqa: E402
from utils.plots import colors  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


def draw_masks(im, boxes, crops, mask_colors, alpha, thick, texts=None):
    """Fill each cropped mask with its color (alpha) and draw its outline, in place."""
    for k, ((x0, y0, x1, y1), crop) in enumerate(zip(boxes, crops)):
        if not crop.size or not crop.any():
            continue
        color = np.array(mask_colors[k], dtype=np.float32)
        region = im[y0:y1, x0:x1]
        region[crop] = (region[crop] * (1 - alpha) + color * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, contours, -1, color.tolist(), thick, offset=(int(x0), int(y0)))
        if texts:
            cv2.putText(im, texts[k], (int(x0), max(int(y0) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick,
                        color.tolist(), thick, cv2.LINE_AA)


def header(width, text, thick):
    """White bar with black text above one panel."""
    scale = 0.6 * thick
    (_, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    bar = np.full((th * 2 + 10, width, 3), 255, dtype=np.uint8)
    cv2.putText(bar, text, (10, th + 5 + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return bar


def side_by_side(im0, gt_cls, gt, pred_cls, pred_conf, pred, names, stats, config):
    """GT panel | gap | prediction panel, each with a header."""
    cp, cfg_plots = config.compare_plots, config.plots
    alpha = float(cfg_plots.alpha)
    h, w = im0.shape[:2]
    thick = max(1, round(max(h, w) / 1000))
    gt_color = [int(v) for v in cfg_plots.gt_color]

    left = im0.copy()
    draw_masks(left, *gt, [gt_color] * len(gt_cls), alpha, thick)
    right = im0.copy()
    texts = [f'{names[int(c)]} {conf:.2f}' for c, conf in zip(pred_cls, pred_conf)] if cfg_plots.show_conf else None
    draw_masks(right, *pred, [colors(int(c), True) for c in pred_cls], alpha, thick, texts)
    if cp.gt_outline_on_pred:  # label outlines on the prediction panel too, to see the boundary error directly
        for (x0, y0, x1, y1), crop in zip(*gt):
            if crop.size and crop.any():
                contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(right, contours, -1, gt_color, thick, offset=(int(x0), int(y0)))

    left = np.vstack([header(w, f"GT: {stats['Labels']} labels", thick), left])
    right = np.vstack([header(w, f"Pred: {stats['Predictions']}  TP {stats['TP']} FP {stats['FP']} FN {stats['FN']}  "
                                 f"IoU {stats['Pixel IoU']:.3f} P {stats['Pixel precision']:.3f} "
                                 f"R {stats['Pixel recall']:.3f}", thick), right])
    gap = np.full((left.shape[0], int(cp.gap), 3), 255, dtype=np.uint8)
    return np.hstack([left, gap, right])


@torch.no_grad()
def main(config):
    cp, pc = config.compare_plots, config.pixel
    weights = cp.get('weights') or config.model.weights
    data_yaml = cp.get('data_yaml') or config.dataset.data_yaml
    if config.model.get('require_cuda', True) and (str(config.model.device).lower() == 'cpu' or
                                                   not torch.cuda.is_available()):
        raise SystemExit(f'Inspection must run on GPU, but model.device is "{config.model.device}" and CUDA '
                         f'available is {torch.cuda.is_available()}. Set model.require_cuda: false to run on CPU.')

    device = select_device(config.model.device)
    half = bool(config.model.half) and device.type != 'cpu'
    model = DetectMultiBackend(weights, device=device, data=data_yaml, fp16=half)
    stride, names = model.stride, model.names
    imgsz = check_img_size(config.model.imgsz, s=stride)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))
    decode_masks = make_mask_decoder(config)

    data = check_dataset(data_yaml)
    files = list_images(data[config.dataset.split])
    if cp.num_images:
        files = files[:int(cp.num_images)]
    labels = img2label_paths([str(f) for f in files])
    classes = list(range(len(names)))
    out_dir = os.path.join(config.dataset.test_output, 'images')
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f'Weights: {weights}\nData: {data_yaml} split {config.dataset.split}, {len(files)} images\n'
                f'Confidence {pc.conf_thres}, NMS IoU {pc.iou_nms}, imgsz {imgsz}, half {half}')

    rows = []
    for i, (file, label) in enumerate(zip(files, labels), 1):
        im0 = cv2.imread(str(file))
        if im0 is None:
            logger.warning(f'Cannot read {file}')
            continue
        h, w = im0.shape[:2]
        im, _, pad = letterbox(im0, imgsz, stride=stride, auto=model.pt)
        im = torch.from_numpy(im.transpose(2, 0, 1)[::-1].copy()).to(device)
        im = (im.half() if half else im.float())[None] / 255
        pred, proto = model(im)[:2]
        det = non_max_suppression(pred, pc.conf_thres, pc.iou_nms, max_det=pc.max_det, nm=32)[0]
        masks = decode_masks(proto[0], det[:, 6:].float(), det[:, :4].float(), im.shape[2:]) if len(det) \
            else det[:0, 0:0]
        pred_inst = to_original(masks, pad, h, w)
        pred_cls = det[:, 5].int().cpu().numpy()
        pred_conf = det[:, 4].float().cpu().numpy()

        gt_cls, *gt_inst = load_gt(label, h, w)
        ious, fp, fn = instance_stats(pred_cls, pred_inst, gt_cls, gt_inst, pc.match_iou)
        a, b, c = pixel_stats(pred_cls, pred_inst, gt_cls, gt_inst, classes, h, w)
        name = f'{i:04d}_{os.path.splitext(os.path.basename(str(file)))[0]}.jpg'
        stats = {'Index': i, 'Plot': name, 'Image': str(file), 'Width': w, 'Height': h, 'Labels': len(gt_cls),
                 'Predictions': len(pred_cls), 'TP': len(ious), 'FP': fp, 'FN': fn,
                 'Pixel IoU': round(mean(a), 6), 'Pixel precision': round(mean(b), 6),
                 'Pixel recall': round(mean(c), 6), 'Mean IoU (matched)': round(mean(ious), 6)}
        rows.append(stats)
        cv2.imwrite(os.path.join(out_dir, name), side_by_side(im0, gt_cls, gt_inst, pred_cls, pred_conf, pred_inst,
                                                              names, stats, config))
        if i % 20 == 0 or i == len(files):
            logger.info(f'{i}/{len(files)} images')

    def badness(r):  # most errors first, then lowest pixel IoU (NaN = empty image, first among ties)
        iou = r['Pixel IoU']
        return -(r['FP'] + r['FN']), iou if iou == iou else -1.0
    write_csv(os.path.join(config.dataset.test_output, 'compare.csv'), sorted(rows, key=badness))
    logger.info(f"Saved {len(rows)} images in {out_dir} and compare.csv (worst first)")
    return rows


if __name__ == '__main__':
    # Load configuration
    config = OmegaConf.load('Config/main_config.yaml')
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(config.environment.cuda_launch_blocking)

    weights = config.compare_plots.get('weights') or config.model.weights
    if not config.get('model_name'):  # default: the training run folder that contains weights/best.pt
        config.model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(weights))))

    # Generate timestamp and run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = os.path.join('bin', f'Run_{timestamp}_{config.model_name}_compare')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Update config paths
    config.dataset.save_path = checkpoints_dir
    config.dataset.test_output = checkpoints_dir

    # Save the config used for this run
    OmegaConf.save(config, os.path.join(checkpoints_dir, "config_used.yaml"))

    prepare_logger(checkpoints_dir, 'compare.log')
    logger.info(f'Compare plots output directory: {checkpoints_dir}')
    main(config)
