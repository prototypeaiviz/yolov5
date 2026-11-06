"""Pixel-level and instance-level IoU, precision and recall for YOLOv5 segmentation.

Usage, from the repo root:
    python segment/eval_iou.py --weights best.pt --data data.yaml --device 0
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.dataloaders import IMG_FORMATS, img2label_paths  # noqa: E402
from utils.general import check_dataset, check_img_size, non_max_suppression  # noqa: E402
from utils.segment.general import process_mask  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


def list_images(val):
    files = []
    for entry in val if isinstance(val, list) else [val]:
        p = Path(entry)
        if p.is_dir():
            files += [f for f in p.rglob('*') if f.suffix[1:].lower() in IMG_FORMATS]
        else:
            files += [p.parent / line.strip() for line in p.read_text().splitlines() if line.strip()]
    return sorted(files)


def load_gt(label_path, h, w):
    """Return class ids (n,) and boolean masks (n, h, w) from a YOLO polygon label file."""
    classes, masks = [], []
    path = Path(label_path)
    if path.exists():
        for line in path.read_text().splitlines():
            values = line.split()
            if len(values) < 7:  # class + at least 3 points
                continue
            pts = np.array(values[1:], dtype=np.float64).reshape(-1, 2) * (w, h)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
            classes.append(int(float(values[0])))
            masks.append(mask.astype(bool))
    if not masks:
        return np.zeros(0, dtype=int), np.zeros((0, h, w), dtype=bool)
    return np.array(classes), np.stack(masks)


def to_original(masks, pad, h, w):
    """Remove letterbox padding and resize masks (n, H, W) to the original image size."""
    if not len(masks):
        return np.zeros((0, h, w), dtype=bool)
    H, W = masks.shape[1:]
    top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))
    bottom, right = H - int(round(pad[1] + 0.1)), W - int(round(pad[0] + 0.1))
    resized = F.interpolate(masks[:, top:bottom, left:right][None].float(), (h, w), mode='nearest')[0]
    return resized.bool().cpu().numpy()


def instance_stats(pred_cls, pred_masks, gt_cls, gt_masks, iou_thres):
    """Greedy matching in descending confidence order. Predictions must already be sorted."""
    used = np.zeros(len(gt_cls), dtype=bool)
    matched_ious = []
    fp = 0
    for cls, mask in zip(pred_cls, pred_masks):
        best_iou, best_j = 0.0, -1
        for j in np.flatnonzero((gt_cls == cls) & ~used):
            union = (mask | gt_masks[j]).sum()
            iou = (mask & gt_masks[j]).sum() / union if union else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thres:
            used[best_j] = True
            matched_ious.append(float(best_iou))
        else:
            fp += 1
    return matched_ious, fp, int((~used).sum())


def pixel_stats(pred_cls, pred_masks, gt_cls, gt_masks, classes):
    """Per-class IoU, precision and recall for one image, using the union of each class's masks."""
    ious, precisions, recalls = [], [], []
    for c in classes:
        pred = pred_masks[pred_cls == c].any(0)
        gt = gt_masks[gt_cls == c].any(0)
        n_pred, n_gt = int(pred.sum()), int(gt.sum())
        if not n_pred and not n_gt:
            continue  # class absent and correctly not predicted
        tp = int((pred & gt).sum())
        ious.append(tp / (n_pred + n_gt - tp))
        if n_pred:
            precisions.append(tp / n_pred)
        if n_gt:
            recalls.append(tp / n_gt)
    return ious, precisions, recalls


def mean(values):
    return float(np.mean(values)) if len(values) else float('nan')


@torch.no_grad()
def main(opt):
    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=device, data=opt.data)
    stride, names = model.stride, model.names
    imgsz = check_img_size(opt.imgsz, s=stride)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))

    data = check_dataset(opt.data)
    files = list_images(data[opt.split])
    labels = img2label_paths([str(f) for f in files])
    classes = list(range(len(names)))
    if not files:
        raise SystemExit(f'No images found for split {opt.split}')

    px_iou, px_precision, px_recall, matched = [], [], [], []
    tp = fp = fn = 0
    for i, (file, label) in enumerate(zip(files, labels), 1):
        im0 = cv2.imread(str(file))
        if im0 is None:
            print(f'WARNING: cannot read {file}')
            continue
        h, w = im0.shape[:2]
        im, _, pad = letterbox(im0, imgsz, stride=stride, auto=model.pt)
        im = torch.from_numpy(im.transpose(2, 0, 1)[::-1].copy()).to(device).float()[None] / 255
        pred, proto = model(im)[:2]
        det = non_max_suppression(pred, opt.conf_thres, opt.iou_nms, max_det=1000, nm=32)[0]
        masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True) if len(det) else det[:0, 0:0]
        pred_masks = to_original(masks, pad, h, w)
        pred_cls = det[:, 5].int().cpu().numpy()
        gt_cls, gt_masks = load_gt(label, h, w)

        ious, fp_i, fn_i = instance_stats(pred_cls, pred_masks, gt_cls, gt_masks, opt.match_iou)
        matched += ious
        tp += len(ious)
        fp += fp_i
        fn += fn_i
        a, b, c = pixel_stats(pred_cls, pred_masks, gt_cls, gt_masks, classes)
        px_iou += a
        px_precision += b
        px_recall += c
        if i % 50 == 0:
            print(f'{i}/{len(files)}')

    results = {
        'images': len(files),
        'conf_thres': opt.conf_thres,
        'pixel': {'iou': mean(px_iou), 'precision': mean(px_precision), 'recall': mean(px_recall)},
        'instance': {
            'mean_iou_matched': mean(matched),
            'precision': tp / (tp + fp) if tp + fp else float('nan'),
            'recall': tp / (tp + fn) if tp + fn else float('nan'),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'match_iou': opt.match_iou}}
    print(json.dumps(results, indent=2))
    if opt.save_json:
        Path(opt.save_json).write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-nms', type=float, default=0.45)
    parser.add_argument('--match-iou', type=float, default=0.5)
    parser.add_argument('--device', default='')
    parser.add_argument('--save-json', default='')
    main(parser.parse_args())
