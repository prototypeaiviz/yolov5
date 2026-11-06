"""One report per trained YOLOv5 segmentation checkpoint: accuracy metrics and timing.

All settings come from Config/main_config.yaml. Run from the repo root after training:
    python segment/model_report.py          # PyTorch (.pt), columns end in "Torch"
    python segment/model_report_onnx.py     # ONNX Runtime, columns end in "ONNX"

Outputs in bin/Run_<timestamp>_<model_name>/:
    config_used.yaml
    test/report.log
    test/report.json    every number, keyed by the report column names
    test/val/           segment/val.py plots

Metric sources:
    Box/Mask mAP*, per-class Mask mAP50-95   segment/val.py at map.conf_thres (standard mAP)
    IoU / Precision / Recall                  pixel-level, segment/eval_iou.py logic at pixel.conf_thres
    Instance precision / recall               greedy mask matching at pixel.match_iou, same confidence
Timing (batch 1, per image, CUDA synchronized):
    Pre    letterbox + HWC->CHW + to device + normalize (image decoding from disk excluded)
    Model  forward pass (for ONNX this includes the host/device copies done by DetectMultiBackend)
    Post   NMS + mask decoding + resize masks to the original image size
"""
from datetime import datetime
import json
import os
import sys
import time

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


def prepare_run(config, suffix=''):
    """Create bin/Run_<timestamp>_<model_name><suffix>/test, save the config, start the logger."""
    if not config.get('model_name'):  # default: the training run folder that contains weights/best.pt
        config.model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(config.model.weights))))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_name = f'Run_{timestamp}_{config.model_name}{suffix}'
    checkpoints_dir = os.path.join('bin', new_run_name)
    os.makedirs(os.path.join(checkpoints_dir, "test"), exist_ok=True)

    config.dataset.save_path = checkpoints_dir
    config.dataset.test_output = os.path.join(checkpoints_dir, "test")
    OmegaConf.save(config, os.path.join(checkpoints_dir, "config_used.yaml"))

    prepare_logger(config.dataset.test_output, 'report.log')
    logger.info(f'Output directory: {config.dataset.test_output}')


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run_map(config, weights, half):
    """Standard mAP evaluation with segment/val.py. Returns box/mask metrics and per-class mask mAP50-95."""
    (mp_b, mr_b, map50_b, map_b, mp_m, mr_m, map50_m, map_m, *_), _, speeds, metrics = seg_val.run(
        data=config.dataset.data_yaml,
        weights=weights,
        batch_size=config.map.batch_size,  # val.py forces 1 for ONNX
        imgsz=config.model.imgsz,
        conf_thres=config.map.conf_thres,
        iou_thres=config.map.iou_thres,
        max_det=config.map.max_det,
        task=config.dataset.split,
        device=config.model.device,
        workers=config.map.workers,
        half=half,
        project=config.dataset.test_output,
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


def run_pixel_and_timing(config, weights, half):
    """Pixel/instance metrics at a fixed confidence, and per-image Pre/Model/Post timing at batch 1."""
    device = select_device(config.model.device)
    half = bool(half) and device.type != 'cpu'
    model = DetectMultiBackend(weights, device=device, data=config.dataset.data_yaml, fp16=half)
    stride, names = model.stride, model.names
    imgsz = check_img_size(config.model.imgsz, s=stride)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))
    session = getattr(model, 'session', None)
    providers = session.get_providers() if session is not None else []

    data = check_dataset(config.dataset.data_yaml)
    files = list_images(data[config.dataset.split])
    if not files:
        raise SystemExit(f'No images found for split {config.dataset.split}')
    labels = img2label_paths([str(f) for f in files])
    classes = list(range(len(names)))
    pc = config.pixel

    def infer(im0):
        """Run one image, return (boxes, cropped masks) at original size, classes, and the three stage times in ms."""
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
        pred_inst = to_original(masks, pad, h, w)
        pred_cls = det[:, 5].int().cpu().numpy()
        sync(device)
        t3 = time.perf_counter()
        return pred_inst, pred_cls, ((t1 - t0) * 1E3, (t2 - t1) * 1E3, (t3 - t2) * 1E3)

    # Warmup on real images, not recorded
    n_warm = min(int(config.timing.warmup_images), len(files))
    for file in files[:n_warm]:
        im0 = cv2.imread(str(file))
        if im0 is not None:
            infer(im0)
    logger.info(f'Timing warmup done on {n_warm} images')

    px_iou, px_precision, px_recall, matched, times = [], [], [], [], []
    tp = fp = fn = 0
    logger.info(f'Pixel metrics and timing on {len(files)} images')
    t_loop = time.perf_counter()
    for i, (file, label) in enumerate(zip(files, labels), 1):
        t_img = time.perf_counter()
        im0 = cv2.imread(str(file))
        if im0 is None:
            logger.warning(f'Cannot read {file}')
            continue
        pred_inst, pred_cls, t = infer(im0)
        times.append(t)

        h, w = im0.shape[:2]
        gt_cls, *gt_inst = load_gt(label, h, w)
        ious, fp_i, fn_i = instance_stats(pred_cls, pred_inst, gt_cls, gt_inst, pc.match_iou)
        matched += ious
        tp += len(ious)
        fp += fp_i
        fn += fn_i
        a, b, c = pixel_stats(pred_cls, pred_inst, gt_cls, gt_inst, classes, h, w)
        px_iou += a
        px_precision += b
        px_recall += c

        dt = time.perf_counter() - t_img
        if dt > 10:
            logger.warning(f'Slow image {i}: {dt:.1f} s, {w}x{h}, {len(pred_cls)} predictions, '
                           f'{len(gt_cls)} labels: {file}')
        if i % 10 == 0 or i == len(files):
            elapsed = time.perf_counter() - t_loop
            logger.info(f'{i}/{len(files)}  {elapsed:.0f} s elapsed, {elapsed / i:.2f} s/image')

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
        'pre_p95_ms': float(np.percentile(times[:, 0], 95)),
        'model_p95_ms': float(np.percentile(times[:, 1], 95)),
        'post_p95_ms': float(np.percentile(times[:, 2], 95)),
        'total_p95_ms': float(np.percentile(total, 95)),
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'providers': providers,
        'half': half}


def main(config, weights=None, label='Torch', half=None, extra=None):
    """Run every metric and timing for one model. `label` is the column suffix (Torch, ONNX)."""
    weights = weights or config.model.weights
    half = config.model.half if half is None else half
    logger.info(f'Weights: {weights}  backend: {label}  half: {half}')
    t_start = time.perf_counter()

    m = run_map(config, weights, half)
    names, p = run_pixel_and_timing(config, weights, half)

    report = {
        f'{label} Pre (ms)': p['pre_ms'],
        f'{label} Model (ms)': p['model_ms'],
        f'{label} Post (ms)': p['post_ms'],
        f'{label} Total Avg (ms)': p['total_avg_ms'],
        f'{label} Total P95 (ms)': p['total_p95_ms'],
        f'Box mAP50 {label}': m['box_map50'],
        f'Box mAP50-95 {label}': m['box_map50_95'],
        f'Mask mAP50 {label}': m['mask_map50'],
        f'Mask mAP75 {label}': m['mask_map75'],
        f'Mask mAP50-95 {label}': m['mask_map50_95'],
        f'IoU {label}': p['pixel_iou'],
        f'Precision {label}': p['pixel_precision'],
        f'Recall {label}': p['pixel_recall']}
    present = dict(zip(m['class_index'], m['class_mask_map50_95']))
    for i, name in names.items():
        report[f'{str(name).capitalize()} Mask mAP50-95 {label}'] = present.get(int(i), 'N/A')

    result = {
        'report': report,
        'extra': {
            **(extra or {}),
            'Instance precision': p['instance_precision'],
            'Instance recall': p['instance_recall'],
            'Instance mean IoU (matched)': p['instance_mean_iou_matched'],
            'Instance TP/FP/FN': [p['tp'], p['fp'], p['fn']],
            f'Box mAP75 {label}': m['box_map75'],
            'val.py Box P/R (best-F1 conf)': [m['box_precision_best_f1'], m['box_recall_best_f1']],
            'val.py Mask P/R (best-F1 conf)': [m['mask_precision_best_f1'], m['mask_recall_best_f1']],
            'val.py speed pre/inference/NMS (ms)': m['val_speed_ms'],
            'P95 pre/model/post (ms)': [p['pre_p95_ms'], p['model_p95_ms'], p['post_p95_ms']],
            'Report wall time (s)': time.perf_counter() - t_start},
        'settings': {
            'backend': label,
            'weights': weights,
            'data': config.dataset.data_yaml,
            'split': config.dataset.split,
            'images_timed': p['images'],
            'device': p['device'],
            'onnxruntime_providers': p['providers'],
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

    prepare_run(config, suffix='_torch')
    main(config)
