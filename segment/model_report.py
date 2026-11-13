"""One report per trained YOLOv5 segmentation checkpoint: accuracy metrics and timing.

All settings come from Config/main_config.yaml. Run from the repo root after training:
    python segment/model_report.py          # PyTorch (.pt), columns end in "Torch"
    python segment/model_report_onnx.py     # ONNX Runtime, columns end in "ONNX"

Outputs in bin/Run_<timestamp>_<model_name>/:
    config_used.yaml
    test/report.log
    test/report.json    every number, keyed by the report column names
    test/report.csv     same numbers as one CSV row
    test/summary.md     readable summary table of this run
    test/sheet.csv      the results-sheet columns in sheet order (accuracy, then Pre/Model/Post); sheet.tsv same,
                        tab-separated for pasting into a spreadsheet
    test/per_image.csv  per-image TP/FP/FN, IoU, precision, recall, sorted worst first
    test/worst_plots/   the plots.num_worst worst images (file name: rank, FP, FN, IoU)
    <report.summary_csv> one row appended per run (Torch and ONNX), for comparing models
    test/val/           segment/val.py plots (masks drawn at 160x160, coarse)
    test/pred_plots/    predictions drawn at full original resolution (plots.num_images images)

Metric sources:
    Box/Mask mAP*, per-class Mask mAP50-95   segment/val.py at map.conf_thres (standard mAP)
    IoU / Precision / Recall                  pixel-level, segment/eval_iou.py logic at pixel.conf_thres
    Instance precision / recall               greedy mask matching at pixel.match_iou, same confidence
Timing (batch 1, per image, CUDA synchronized):
    Pre    letterbox + HWC->CHW + to device + normalize (image decoding from disk excluded)
    Model  forward pass (for ONNX this includes the host/device copies done by DetectMultiBackend)
    Post   NMS + mask decoding on the device (masks at network input size, pixel.mask_mode decides how)
    Mapping masks to the original image size is evaluation-only work and is reported separately in "extra".
"""
from datetime import datetime
import csv
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
from utils.plots import colors  # noqa: E402
from utils.segment.general import process_mask, process_mask_upsample  # noqa: E402
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
        batch_size=1,  # always batch 1 for validation testing
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
        return_metrics=True,
        upsample_masks=config.pixel.get('mask_mode', 'crop_then_upsample') == 'upsample_then_crop',
        mask_thres=float(config.pixel.get('mask_thres', 0.5)))
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


def plot_prediction(im0, pred, pred_cls, pred_conf, gt, names, cfg_plots, path):
    """Draw predicted masks at the original image resolution, optionally with label outlines, and save as JPEG.

    Predictions: filled with the class color plus a class-color outline. Labels: thin outline in plots.gt_color (BGR).
    """
    canvas = im0.copy()
    alpha = float(cfg_plots.alpha)
    thick = max(1, round(max(im0.shape[:2]) / 1000))
    for (x0, y0, x1, y1), crop, c, conf in zip(*pred, pred_cls, pred_conf):
        if not crop.any():
            continue
        color = np.array(colors(int(c), bgr=True), dtype=np.float32)
        region = canvas[y0:y1, x0:x1]
        region[crop] = (region[crop] * (1 - alpha) + color * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, color.tolist(), thick, offset=(int(x0), int(y0)))
        if cfg_plots.show_conf:
            cv2.putText(canvas, f'{names[int(c)]} {conf:.2f}', (int(x0), max(int(y0) - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick, color.tolist(), thick, cv2.LINE_AA)
    if cfg_plots.draw_gt:
        gt_color = tuple(int(v) for v in cfg_plots.gt_color)
        for (x0, y0, x1, y1), crop in zip(*gt):
            if crop.size:
                contours, _ = cv2.findContours(crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(canvas, contours, -1, gt_color, thick, offset=(int(x0), int(y0)))
    cv2.imwrite(path, canvas)


# List values in "extra" that are split into one CSV column per element
CSV_SPLIT = {
    'Instance TP/FP/FN': ['Instance TP', 'Instance FP', 'Instance FN'],
    'val.py Box P/R (best-F1 conf)': ['val.py Box P (best-F1 conf)', 'val.py Box R (best-F1 conf)'],
    'val.py Mask P/R (best-F1 conf)': ['val.py Mask P (best-F1 conf)', 'val.py Mask R (best-F1 conf)'],
    'val.py speed pre/inference/NMS (ms)': ['val.py speed pre (ms)', 'val.py speed inference (ms)',
                                            'val.py speed NMS (ms)'],
    'P95 pre/model/post/total (ms)': ['P95 pre (ms)', 'P95 model (ms)', 'P95 post (ms)', 'P95 total (ms)'],
    'Mask to original size avg/P95 (ms, eval only)': ['Mask to original size avg (ms, eval only)',
                                                      'Mask to original size P95 (ms, eval only)']}


def csv_row(config, result):
    """Flatten report, extra and settings into one CSV row (ordered dict of column -> value)."""
    row = {'Run': os.path.basename(config.dataset.save_path),
           'Model': config.model_name,
           'Backend': result['settings']['backend'],
           'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    for section in ('report', 'extra', 'settings'):
        for k, v in result[section].items():
            if isinstance(v, (list, tuple)):
                names = CSV_SPLIT.get(k)
                if names and len(names) == len(v):
                    row.update(zip(names, v))
                else:
                    row[k] = ';'.join(str(x) for x in v)
            else:
                row[k] = v
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()}


def sheet_row(report, label):
    """The columns of the results sheet, in the sheet's order: accuracy on the val set, then inspection time."""
    cols = [f'Box mAP50 {label}', f'Box mAP50-95 {label}', f'Mask mAP50 {label}', f'Mask mAP50-95 {label}',
            f'Mask mAP75 {label}', f'IoU {label}', f'Precision {label}', f'Recall {label}']
    cols += [k for k in report if k.endswith(f' Mask mAP50-95 {label}') and k != f'Mask mAP50-95 {label}']  # classes
    cols += [f'{label} Pre (ms)', f'{label} Model (ms)', f'{label} Post (ms)']
    row = {}
    for k in cols:
        v = report[k]
        row[k] = round(v, 2 if k.endswith('(ms)') else 4) if isinstance(v, float) else v
    return row


def write_summary(path, config, result):
    """Human-readable summary of one run: report columns in sheet order, then extras and settings."""
    def fmt(v):
        if isinstance(v, float):
            return f'{v:.4f}'
        if isinstance(v, (list, tuple)):
            return ', '.join(fmt(x) for x in v)
        return str(v)

    st = result['settings']
    lines = [f"# {os.path.basename(config.dataset.save_path)}", '',
             f"- Model: {config.model_name}  ({st['backend']}, {'FP16' if st['half'] else 'FP32'}, {st['device']})",
             f"- Weights: {st['weights']}",
             f"- Data: {st['data']}  split {st['split']}, {st['images_timed']} images",
             f"- Confidence: mAP {st['map_conf_thres']}, IoU/Precision/Recall {st['pixel_conf_thres']}, "
             f"mask_mode {st['mask_mode']}, mask threshold {st.get('mask_thres', 0.5)}",
             '', '## Report', '', '| Column | Value |', '|---|---|']
    lines += [f'| {k} | {fmt(v)} |' for k, v in result['report'].items()]
    lines += ['', '## Extra', '', '| Item | Value |', '|---|---|']
    lines += [f'| {k} | {fmt(v)} |' for k, v in result['extra'].items()]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_csv(path, rows, delimiter=','):
    """Write rows with the union of their columns, in first-seen order."""
    columns = []
    for r in rows:
        columns += [k for k in r if k not in columns]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval='', delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def append_summary(path, row):
    """Append one row to the summary CSV. Rewrites the file so new columns (other backend/classes) are added."""
    rows = []
    if os.path.exists(path):
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    write_csv(path, rows + [row])


def make_mask_decoder(config):
    """Mask decoding function from pixel.mask_mode and pixel.mask_thres: (protos, coefficients, boxes, shape) -> masks
    at network input size."""
    mask_mode = config.pixel.get('mask_mode', 'crop_then_upsample')
    mask_thres = float(config.pixel.get('mask_thres', 0.5))
    if mask_mode == 'upsample_then_crop':  # upsample masks to input size, then cut at the box: smoother edges
        def decode_masks(protos, masks_in, bboxes, shape):
            return process_mask_upsample(protos, masks_in, bboxes, shape, thres=mask_thres)
    elif mask_mode == 'crop_then_upsample':  # cut at the box on the 160x160 grid, then upsample (YOLOv5 predict.py)
        def decode_masks(protos, masks_in, bboxes, shape):
            return process_mask(protos, masks_in, bboxes, shape, upsample=True, thres=mask_thres)
    else:
        raise SystemExit(f'Unknown pixel.mask_mode "{mask_mode}", use upsample_then_crop or crop_then_upsample')
    logger.info(f'Mask post-processing: {mask_mode}, mask threshold {mask_thres}')
    return decode_masks


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

    decode_masks = make_mask_decoder(config)

    def infer(im0):
        """Run one image. Return (boxes, cropped masks) at original size, classes, confidences and stage times in ms:
        pre, model, post (NMS + mask decoding), mask-to-original (evaluation only, not part of the total)."""
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
        det = non_max_suppression(pred, pc.conf_thres, pc.iou_nms, max_det=pc.max_det, nm=proto.shape[1])[0]
        masks = decode_masks(proto[0], det[:, 6:].float(), det[:, :4].float(), im.shape[2:]) if len(det) else det[:0, 0:0]
        sync(device)
        t3 = time.perf_counter()
        pred_inst = to_original(masks, pad, h, w)
        pred_cls = det[:, 5].int().cpu().numpy()
        pred_conf = det[:, 4].float().cpu().numpy()
        t4 = time.perf_counter()
        return pred_inst, pred_cls, pred_conf, ((t1 - t0) * 1E3, (t2 - t1) * 1E3, (t3 - t2) * 1E3, (t4 - t3) * 1E3)

    # Warmup on real images, not recorded
    n_warm = min(int(config.timing.warmup_images), len(files))
    for file in files[:n_warm]:
        im0 = cv2.imread(str(file))
        if im0 is not None:
            infer(im0)
    logger.info(f'Timing warmup done on {n_warm} images')

    cfg_plots = config.plots
    n_plots = int(cfg_plots.num_images)
    plot_dir = os.path.join(config.dataset.test_output, 'pred_plots')
    if n_plots:
        os.makedirs(plot_dir, exist_ok=True)

    px_iou, px_precision, px_recall, matched, times, per_image = [], [], [], [], [], []
    tp = fp = fn = 0
    logger.info(f'Pixel metrics and timing on {len(files)} images')
    t_loop = time.perf_counter()
    for i, (file, label) in enumerate(zip(files, labels), 1):
        t_img = time.perf_counter()
        im0 = cv2.imread(str(file))
        if im0 is None:
            logger.warning(f'Cannot read {file}')
            continue
        pred_inst, pred_cls, pred_conf, t = infer(im0)
        times.append(t)

        h, w = im0.shape[:2]
        gt_cls, *gt_inst = load_gt(label, h, w)
        if i <= n_plots:
            plot_prediction(im0, pred_inst, pred_cls, pred_conf, gt_inst, names, cfg_plots,
                            os.path.join(plot_dir, f'{i:04d}_{os.path.splitext(os.path.basename(str(file)))[0]}.jpg'))
        ious, fp_i, fn_i = instance_stats(pred_cls, pred_inst, gt_cls, gt_inst, pc.match_iou)
        matched += ious
        tp += len(ious)
        fp += fp_i
        fn += fn_i
        a, b, c = pixel_stats(pred_cls, pred_inst, gt_cls, gt_inst, classes, h, w)
        px_iou += a
        px_precision += b
        px_recall += c
        per_image.append({
            'Index': i,
            'Image': str(file),
            'Width': w,
            'Height': h,
            'Labels': len(gt_cls),
            'Predictions': len(pred_cls),
            'TP': len(ious),
            'FP': fp_i,
            'FN': fn_i,
            'Pixel IoU': round(mean(a), 6),
            'Pixel precision': round(mean(b), 6),
            'Pixel recall': round(mean(c), 6),
            'Mean IoU (matched)': round(mean(ious), 6),
            'Min confidence': round(float(pred_conf.min()), 4) if len(pred_conf) else '',
            'Total ms': round(sum(t[:3]), 3)})

        dt = time.perf_counter() - t_img
        if dt > 10:
            logger.warning(f'Slow image {i}: {dt:.1f} s, {w}x{h}, {len(pred_cls)} predictions, '
                           f'{len(gt_cls)} labels: {file}')
        if i % 10 == 0 or i == len(files):
            elapsed = time.perf_counter() - t_loop
            logger.info(f'{i}/{len(files)}  {elapsed:.0f} s elapsed, {elapsed / i:.2f} s/image')

    # Per-image table, worst first: most errors (FP + FN), then lowest pixel IoU
    def badness(r):
        iou = r['Pixel IoU']
        return -(r['FP'] + r['FN']), iou if iou == iou else -1.0  # NaN IoU (empty image) sorts first among ties
    ranked = sorted(per_image, key=badness)
    for rank, r in enumerate(ranked, 1):
        r['Worst rank'] = rank
    write_csv(os.path.join(config.dataset.test_output, 'per_image.csv'), ranked)
    logger.info(f"Saved per-image results: {os.path.join(config.dataset.test_output, 'per_image.csv')}")

    # Plots of the worst images (inference again, not timed)
    n_worst = int(cfg_plots.get('num_worst', 0))
    if n_worst:
        worst_dir = os.path.join(config.dataset.test_output, 'worst_plots')
        os.makedirs(worst_dir, exist_ok=True)
        for r in ranked[:n_worst]:
            im0 = cv2.imread(r['Image'])
            pred_inst, pred_cls, pred_conf, _ = infer(im0)
            _, *gt_inst = load_gt(labels[r['Index'] - 1], *im0.shape[:2])
            name = (f"{r['Worst rank']:03d}_FP{r['FP']}_FN{r['FN']}_IoU{r['Pixel IoU']:.3f}_"
                    f"{os.path.splitext(os.path.basename(r['Image']))[0]}.jpg")
            plot_prediction(im0, pred_inst, pred_cls, pred_conf, gt_inst, names, cfg_plots,
                            os.path.join(worst_dir, name))
        logger.info(f'Saved {min(n_worst, len(ranked))} worst-image plots to {worst_dir}')

    times = np.array(times)  # (n, 4) pre, model, post, mask-to-original
    total = times[:, :3].sum(1)  # deployment total: pre + model + post
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
        'to_original_ms': float(times[:, 3].mean()),
        'to_original_p95_ms': float(np.percentile(times[:, 3], 95)),
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'providers': providers,
        'half': half}


def main(config, weights=None, label='Torch', half=None, extra=None):
    """Run every metric and timing for one model. `label` is the column suffix (Torch, ONNX)."""
    weights = weights or config.model.weights
    half = config.model.half if half is None else half
    if config.model.get('require_cuda', True) and (str(config.model.device).lower() == 'cpu' or
                                                   not torch.cuda.is_available()):
        raise SystemExit(f'Inspection must run on GPU, but model.device is "{config.model.device}" and CUDA '
                         f'available is {torch.cuda.is_available()}. Set model.require_cuda: false to run on CPU.')
    logger.info(f'Weights: {weights}  backend: {label}  half: {half}')
    t_start = time.perf_counter()

    m = run_map(config, weights, half)
    names, p = run_pixel_and_timing(config, weights, half)

    report = {
        f'{label} Pre (ms)': p['pre_ms'],
        f'{label} Model (ms)': p['model_ms'],
        f'{label} Post (ms)': p['post_ms'],
        f'{label} Total Avg (ms)': p['pre_ms'] + p['model_ms'] + p['post_ms'],  # Pre + Model + Post
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
            'P95 pre/model/post/total (ms)': [p['pre_p95_ms'], p['model_p95_ms'], p['post_p95_ms'], p['total_p95_ms']],
            'Mask to original size avg/P95 (ms, eval only)': [p['to_original_ms'], p['to_original_p95_ms']],
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
            'mask_mode': config.pixel.get('mask_mode', 'crop_then_upsample'),
            'mask_thres': float(config.pixel.get('mask_thres', 0.5)),
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

    summary_md = os.path.join(config.dataset.test_output, 'summary.md')
    write_summary(summary_md, config, result)
    logger.info(f'Saved {summary_md}')

    result['sheet'] = sheet_row(result['report'], result['settings']['backend'])
    for name, delim in (('sheet.csv', ','), ('sheet.tsv', '\t')):  # .tsv pastes straight into spreadsheet cells
        write_csv(os.path.join(config.dataset.test_output, name), [result['sheet']], delimiter=delim)
    logger.info(f"Saved sheet.csv / sheet.tsv in {config.dataset.test_output}")

    row = csv_row(config, result)
    csv_path = os.path.join(config.dataset.test_output, 'report.csv')
    write_csv(csv_path, [row])
    logger.info(f'Saved {csv_path}')
    summary = config.get('report', {}).get('summary_csv', '')
    if summary:
        append_summary(summary, row)
        logger.info(f'Appended to {summary}')
    return result


if __name__ == '__main__':
    # Load configuration
    config = OmegaConf.load('Config/main_config.yaml')
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(config.environment.cuda_launch_blocking)

    prepare_run(config, suffix='_torch')
    main(config)
