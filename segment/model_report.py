"""One report per trained YOLOv5 segmentation checkpoint: accuracy metrics and Torch timing.

All settings come from Config/main_config.yaml. Run from the repo root:
    python segment/model_report.py

Outputs in bin/Run_<timestamp>_<model_name>/:
    config_used.yaml
    test/report.log
    test/report.json    every number, keyed by the report column names
    test/val/           segment/val.py plots

Metric sources:
    Box/Mask mAP*, per-class Mask mAP50-95   segment/val.py at map.conf_thres (standard mAP)
    IoU / Precision / Recall Torch            pixel-level, segment/eval_iou.py logic at pixel.conf_thres
    Instance precision / recall               greedy mask matching at pixel.match_iou, same confidence
Timing (batch 1, per image, CUDA synchronized):
    Pre    letterbox + HWC->CHW + to device + normalize (image decoding from disk excluded)
    Model  forward pass
    Post   NMS + mask decoding + resize masks to the original image size
"""
from datetime import datetime
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.common import DetectMultiBackend  # noqa: E402
from segment import val as seg_val  # noqa: E402
from segment.eval_iou import instance_stats, list_images, load_gt, mean, pixel_stats, to_original  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.dataloaders import img2label_paths  # noqa: E402
from utils.general import check_dataset, check_img_size, non_max_suppression  # noqa: E402
from utils.segment.general import process_mask  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


def prepare_logger(log_dir, log_filename):
    """Configure loguru logger with file output"""
    log_path = os.path.join(log_dir, log_filename)
    logger.remove()  # Remove default handler
    logger.add(log_path, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
    logger.add(lambda msg: print(msg, end=""))  # Keep console output


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_map(config):
    """Standard mAP evaluation with segment/val.py. Returns box/mask metrics and per-class mask mAP50-95."""
    out_dir = config.dataset.test_output
    (mp_b, mr_b, map50_b, map_b, mp_m, mr_m, map50_m, map_m, *_), _, speeds, metrics = seg_val.run(
        data=config.dataset.data_yaml,
        weights=config.model.weights,
        batch_size=config.map.batch_size,
        imgsz=config.model.imgsz,
        conf_thres=config.map.conf_thres,
        iou_thres=config.map.iou_thres,
        max_det=config.map.max_det,
        task=config.dataset.split,
        device=config.model.device,
        workers=config.map.workers,
        half=config.model.half,
        project=out_dir,
        name='val',
        exist_ok=True,
        return_metrics=True)
    return {
        'box_map50': float(map50_b),
        'box_map50_95': float(map_b),
        'box_map75': float(metrics.metric_box.map75),
        'mask_map50': float(map50_m),
        'mask_map50_95': float(map_m),
        'mask_map75': float(metrics.metric_mask.map75),
        'box_precision_best_f1': float(mp_b),
        'box_recall_best_f1': float(mr_b),
        'mask_precision_best_f1': float(mp_m),
        'mask_recall_best_f1': float(mr_m),
        'class_index': [int(c) for c in metrics.ap_class_index],
        'class_mask_map50_95': [float(v) for v in metrics.metric_mask.ap],
        'val_speed_ms': [float(v) for v in speeds]}


def run_pixel_and_timing(config):
    """Pixel/instance metrics at a fixed confidence, and per-image Pre/Model/Post timing at batch 1."""
    device = select_device(config.model.device)
    half = bool(config.model.half) and device.type != 'cpu'
    model = DetectMultiBackend(config.model.weights, device=device, data=config.dataset.data_yaml, fp16=half)
    stride, names = model.stride, model.names
    imgsz = check_img_size(config.model.imgsz, s=stride)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))

    data = check_dataset(config.dataset.data_yaml)
    files = list_images(data[config.dataset.split])
    if not files:
        raise SystemExit(f'No images found for split {config.dataset.split}')
    labels = img2label_paths([str(f) for f in files])
    classes = list(range(len(names)))
    pc = config.pixel

    def infer(im0):
        """Run one image, return masks at original size, classes, and the three stage times in ms."""
        h, w = im0.shape[:2]
        sync(device)
        t0 = time.perf_counter()
        im, _, pad = letterbox(im0, imgsz, stride=stride, auto=model.pt)
        im = torch.from_numpy(im.transpose(2, 0, 1)[::-1].copy()).to(device)
        im = (im.half() if half else im.float())[None] / 255
        sync(device)
        t1 = time.perf_counter()
        pred, proto = model(im)[:2]
        sync(device)
        t2 = time.perf_counter()
        det = non_max_suppression(pred, pc.conf_thres, pc.iou_nms, max_det=pc.max_det, nm=32)[0]
        masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True) if len(det) else det[:0, 0:0]
        pred_masks = to_original(masks, pad, h, w)
        pred_cls = det[:, 5].int().cpu().numpy()
        sync(device)
        t3 = time.perf_counter()
        return pred_masks, pred_cls, ((t1 - t0) * 1E3, (t2 - t1) * 1E3, (t3 - t2) * 1E3)

    # Warmup on real images, not recorded
    n_warm = min(int(config.timing.warmup_images), len(files))
    for file in files[:n_warm]:
        im0 = cv2.imread(str(file))
        if im0 is not None:
            infer(im0)
    logger.info(f'Timing warmup done on {n_warm} images')

    px_iou, px_precision, px_recall, matched, times = [], [], [], [], []
    tp = fp = fn = 0
    for i, (file, label) in enumerate(zip(files, labels), 1):
        im0 = cv2.imread(str(file))
        if im0 is None:
            logger.warning(f'Cannot read {file}')
            continue
        pred_masks, pred_cls, t = infer(im0)
        times.append(t)

        gt_cls, gt_masks = load_gt(label, *im0.shape[:2])
        ious, fp_i, fn_i = instance_stats(pred_cls, pred_masks, gt_cls, gt_masks, pc.match_iou)
        matched += ious
        tp += len(ious)
        fp += fp_i
        fn += fn_i
        a, b, c = pixel_stats(pred_cls, pred_masks, gt_cls, gt_masks, classes)
        px_iou += a
        px_precision += b
        px_recall += c
        if i % 50 == 0:
            logger.info(f'{i}/{len(files)}')

    times = np.array(times)  # (n, 3) pre, model, post
    total = times.sum(1)
    return names, {
        'images': len(times),
        'pixel_iou': mean(px_iou),
        'pixel_precision': mean(px_precision),
        'pixel_recall': mean(px_recall),
        'instance_precision': tp / (tp + fp) if tp + fp else float('nan'),
        'instance_recall': tp / (tp + fn) if tp + fn else float('nan'),
        'instance_mean_iou_matched': mean(matched),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'pre_ms': float(times[:, 0].mean()),
        'model_ms': float(times[:, 1].mean()),
        'post_ms': float(times[:, 2].mean()),
        'total_avg_ms': float(total.mean()),
        'total_p95_ms': float(np.percentile(total, 95)),
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'half': half}


def main(config):
    m = run_map(config)
    names, p = run_pixel_and_timing(config)

    report = {
        'Torch Pre (ms)': p['pre_ms'],
        'Torch Model (ms)': p['model_ms'],
        'Torch Post (ms)': p['post_ms'],
        'Torch Total Avg (ms)': p['total_avg_ms'],
        'Torch Total P95 (ms)': p['total_p95_ms'],
        'Box mAP50 Torch': m['box_map50'],
        'Box mAP50-95 Torch': m['box_map50_95'],
        'Mask mAP50 Torch': m['mask_map50'],
        'Mask mAP75 Torch': m['mask_map75'],
        'Mask mAP50-95 Torch': m['mask_map50_95'],
        'IoU Torch': p['pixel_iou'],
        'Precision Torch': p['pixel_precision'],
        'Recall Torch': p['pixel_recall']}
    present = dict(zip(m['class_index'], m['class_mask_map50_95']))
    for i, name in names.items():
        report[f'{str(name).capitalize()} Mask mAP50-95 Torch'] = present.get(int(i), 'N/A')

    result = {
        'report': report,
        'extra': {
            'Instance precision': p['instance_precision'],
            'Instance recall': p['instance_recall'],
            'Instance mean IoU (matched)': p['instance_mean_iou_matched'],
            'Instance TP/FP/FN': [p['tp'], p['fp'], p['fn']],
            'Box mAP75 Torch': m['box_map75'],
            'val.py Box P/R (best-F1 conf)': [m['box_precision_best_f1'], m['box_recall_best_f1']],
            'val.py Mask P/R (best-F1 conf)': [m['mask_precision_best_f1'], m['mask_recall_best_f1']],
            'val.py speed pre/inference/NMS (ms, batched)': m['val_speed_ms']},
        'settings': {
            'weights': config.model.weights,
            'data': config.dataset.data_yaml,
            'split': config.dataset.split,
            'images_timed': p['images'],
            'device': p['device'],
            'half': p['half'],
            'imgsz': config.model.imgsz,
            'map_conf_thres': config.map.conf_thres,
            'pixel_conf_thres': config.pixel.conf_thres,
            'match_iou': config.pixel.match_iou}}

    logger.info('Report')
    for k, v in report.items():
        logger.info(f'  {k:<32} {v:.4f}' if isinstance(v, float) else f'  {k:<32} {v}')
    logger.info('Extra')
    for k, v in result['extra'].items():
        logger.info(f'  {k:<44} {v}')

    json_path = os.path.join(config.dataset.test_output, 'report.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f'Saved {json_path}')
    return result


if __name__ == '__main__':
    # Load configuration
    config = OmegaConf.load('Config/main_config.yaml')
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(config.environment.cuda_launch_blocking)

    # Default model name: the training run folder that contains weights/best.pt
    if not config.get('model_name'):
        config.model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(config.model.weights))))

    # Generate timestamp and run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_name = f'Run_{timestamp}_{config.model_name}'
    checkpoints_dir = os.path.join('bin', new_run_name)
    os.makedirs(os.path.join(checkpoints_dir, "test"), exist_ok=True)

    # Update config paths
    config.dataset.save_path = checkpoints_dir
    config.dataset.test_output = os.path.join(checkpoints_dir, "test")

    # Save the config used for this run
    OmegaConf.save(config, os.path.join(checkpoints_dir, "config_used.yaml"))

    prepare_logger(config.dataset.test_output, 'report.log')
    logger.info(f'Weights: {config.model.weights}')
    logger.info(f'Output directory: {config.dataset.test_output}')
    main(config)
