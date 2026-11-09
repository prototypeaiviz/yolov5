"""Stand-alone YOLOv5-seg inference with ONNX Runtime: every pre- and post-processing step written out with numpy
and OpenCV only (no YOLOv5 code), so the same steps can be ported to the inspection software.

Export the model first (fixed 640x640 input, batch 1):
    python export.py --weights <run>/weights/best.pt --include onnx --imgsz 640 --opset 12 --device 0

Settings: onnx_infer section of Config/main_config.yaml. Run from the repo root:
    python segment/onnx_infer.py

Model outputs (static export at 640):
    output0 (1, 25200, 5 + nc + 32)  per anchor: cx, cy, w, h (input pixels), objectness, nc class scores, 32 mask coefs
    output1 (1, 32, 160, 160)        mask prototypes at 1/4 of the input size

Steps per image:
    Pre    1. letterbox: resize keeping the aspect ratio so the long side is 640, pad to 640x640 with gray (114)
           2. BGR -> RGB, HWC -> CHW, uint8 -> float32 / 255, add the batch dimension
    Model  3. session.run
    Post   4. confidence = objectness x class score, keep >= conf_thres
           5. boxes cx,cy,w,h -> x1,y1,x2,y2, NMS per class (iou_nms), keep at most max_det
           6. masks: sigmoid(coefs @ prototypes) -> 160x160, bilinear resize to 640x640, zero outside the box,
              threshold (mask_thres)
           7. back to the original image: remove the padding, resize masks (nearest) and scale boxes

Outputs in bin/Run_<timestamp>_<model_name>_onnx_infer/: drawn predictions (images/), detections.csv, timing.csv.
"""
from datetime import datetime
import csv
import glob
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger
from omegaconf import OmegaConf

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def prepare_logger(log_dir, log_filename):
    """Configure loguru logger with file output"""
    log_path = os.path.join(log_dir, log_filename)
    logger.remove()  # Remove default handler
    logger.add(log_path, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
    logger.add(lambda msg: print(msg, end=""))  # Keep console output


# ----------------------------------------------------------------------------------------------------- pre-processing
def letterbox(im, size, color=(114, 114, 114)):
    """Resize keeping the aspect ratio (long side = size), then pad to size x size, image centered.
    Returns the padded image, the scale factor and the (left, top) padding. Same as YOLOv5 letterbox(auto=False)."""
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    if (new_w, new_h) != (w, h):
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def preprocess(im0, size, dtype):
    """Steps 1-2: letterbox, BGR->RGB, HWC->CHW, /255, batch dimension."""
    im, r, pad = letterbox(im0, size)
    x = im[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    x = np.ascontiguousarray(x, dtype=dtype)[None] / 255.0
    return x.astype(dtype), r, pad


# ---------------------------------------------------------------------------------------------------- post-processing
def nms(boxes, scores, iou_thres):
    """Greedy NMS (same rule as torchvision.ops.nms). boxes (n, 4) x1y1x2y2. Returns kept indices, best first."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thres]
    return np.array(keep, dtype=int)


def detect(out0, nc, conf_thres, iou_thres, max_det, max_wh=7680):
    """Steps 4-5 (as YOLOv5 non_max_suppression, one label per box). Returns boxes (n, 4) x1y1x2y2 in input
    pixels, confidences (n,), classes (n,), mask coefficients (n, 32)."""
    p = out0[0].astype(np.float32)
    p = p[p[:, 4] > conf_thres]  # objectness first (cheap)
    scores = p[:, 5:5 + nc] * p[:, 4:5]  # conf = obj x cls
    cls = scores.argmax(1)
    conf = scores[np.arange(len(p)), cls]
    keep = conf > conf_thres
    p, cls, conf = p[keep], cls[keep], conf[keep]
    if not len(p):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int), np.zeros((0, 32), np.float32)
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    order = conf.argsort()[::-1][:30000]  # same cap as YOLOv5 (max_nms)
    boxes, conf, cls, coefs = boxes[order], conf[order], cls[order], p[order, 5 + nc:]
    k = nms(boxes + cls[:, None] * max_wh, conf, iou_thres)[:max_det]  # class offset = NMS per class
    return boxes[k], conf[k], cls[k], coefs[k]


def decode_masks(coefs, protos, boxes, size, mask_thres):
    """Step 6: masks at input size (n, size, size), bool. Upsample first, then cut at the box (sharper edges)."""
    c, mh, mw = protos.shape
    m = 1 / (1 + np.exp(-(coefs @ protos.reshape(c, -1))))  # sigmoid, (n, mh*mw)
    m = m.reshape(-1, mh, mw)
    masks = np.zeros((len(m), size, size), dtype=bool)
    for k in range(len(m)):
        up = cv2.resize(m[k], (size, size), interpolation=cv2.INTER_LINEAR)  # bilinear, like F.interpolate
        x1, y1, x2, y2 = boxes[k]
        rows, cols = np.arange(size)[:, None], np.arange(size)[None, :]
        inside = (cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2)  # same rule as YOLOv5 crop_mask
        masks[k] = (up > mask_thres) & inside
    return masks


def to_original(masks, boxes, r, pad, h, w):
    """Step 7: remove the padding, resize masks to the original image (nearest), scale and clip boxes."""
    left, top = pad
    new_h, new_w = int(round(h * r)), int(round(w * r))
    out = np.zeros((len(masks), h, w), dtype=bool)
    for k, m in enumerate(masks):
        crop = m[top:top + new_h, left:left + new_w].astype(np.uint8)
        out[k] = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    b = boxes.copy()
    b[:, [0, 2]] = ((b[:, [0, 2]] - left) / r).clip(0, w)
    b[:, [1, 3]] = ((b[:, [1, 3]] - top) / r).clip(0, h)
    return out, b


# ------------------------------------------------------------------------------------------------------------- drawing
def draw(im0, masks, boxes, conf, cls, names, alpha, show_conf):
    im = im0.copy()
    thick = max(1, round(max(im.shape[:2]) / 1000))
    palette = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72)]
    for m, b, s, c in zip(masks, boxes, conf, cls):
        color = np.array(palette[int(c) % len(palette)], dtype=np.float32)
        im[m] = (im[m] * (1 - alpha) + color * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, contours, -1, color.tolist(), thick)
        if show_conf:
            cv2.putText(im, f'{names.get(int(c), int(c))} {s:.2f}', (int(b[0]), max(int(b[1]) - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick, color.tolist(), thick, cv2.LINE_AA)
    return im


def main(config):
    oi = config.onnx_infer
    if not os.path.exists(oi.weights):
        raise SystemExit(f'ONNX model not found: {oi.weights}. Export it first with export.py (see the docstring).')

    gpu = str(oi.device).lower() != 'cpu'
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
    session = ort.InferenceSession(oi.weights, providers=providers)
    active = session.get_providers()
    logger.info(f'onnxruntime {ort.__version__}, active providers {active}')
    if gpu and 'CUDAExecutionProvider' not in active and oi.require_cuda:
        raise SystemExit('GPU requested but ONNX Runtime is not using CUDAExecutionProvider.')

    inp = session.get_inputs()[0]
    size = int(inp.shape[2])  # fixed at export
    dtype = np.float16 if 'float16' in inp.type else np.float32
    meta = session.get_modelmeta().custom_metadata_map  # export.py stores stride and names
    names = eval(meta['names']) if 'names' in meta else {}
    nc = session.get_outputs()[0].shape[2] - 5 - 32
    logger.info(f'Input {inp.name} {inp.shape} {inp.type}, classes {nc} {names}')

    files = sorted(f for f in glob.glob(os.path.join(oi.images, '*')) if f.lower().endswith(IMG_EXT))
    if oi.num_images:
        files = files[:int(oi.num_images)]
    if not files:
        raise SystemExit(f'No images in {oi.images}')
    out_dir = os.path.join(config.dataset.test_output, 'images')
    os.makedirs(out_dir, exist_ok=True)

    # Warmup (first runs allocate GPU memory and pick kernels), not timed
    dummy = np.zeros(inp.shape, dtype=dtype)
    for _ in range(int(oi.warmup)):
        session.run(None, {inp.name: dummy})

    det_rows, time_rows = [], []
    for i, f in enumerate(files, 1):
        im0 = cv2.imread(f)  # BGR, decoding not timed
        if im0 is None:
            logger.warning(f'Cannot read {f}')
            continue
        h, w = im0.shape[:2]
        t0 = time.perf_counter()
        x, r, pad = preprocess(im0, size, dtype)                                              # Pre
        t1 = time.perf_counter()
        out0, protos = session.run(None, {inp.name: x})                                       # Model
        t2 = time.perf_counter()
        boxes, conf, cls, coefs = detect(out0, nc, oi.conf_thres, oi.iou_nms, oi.max_det)     # Post
        masks = decode_masks(coefs, protos[0].astype(np.float32), boxes, size, oi.mask_thres)
        masks, boxes0 = to_original(masks, boxes, r, pad, h, w)
        t3 = time.perf_counter()

        time_rows.append({'Image': os.path.basename(f), 'Width': w, 'Height': h, 'Detections': len(conf),
                          'Pre (ms)': round((t1 - t0) * 1E3, 3), 'Model (ms)': round((t2 - t1) * 1E3, 3),
                          'Post (ms)': round((t3 - t2) * 1E3, 3), 'Total (ms)': round((t3 - t0) * 1E3, 3)})
        for b, s, c, m in zip(boxes0, conf, cls, masks):
            det_rows.append({'Image': os.path.basename(f), 'Class': int(c), 'Name': names.get(int(c), ''),
                             'Confidence': round(float(s), 4), 'x1': round(float(b[0]), 1),
                             'y1': round(float(b[1]), 1), 'x2': round(float(b[2]), 1), 'y2': round(float(b[3]), 1),
                             'Mask pixels': int(m.sum())})
        if oi.save_images:
            cv2.imwrite(os.path.join(out_dir, os.path.basename(f)),
                        draw(im0, masks, boxes0, conf, cls, names, float(oi.alpha), bool(oi.show_conf)))
        if i % 20 == 0 or i == len(files):
            logger.info(f'{i}/{len(files)} images')

    for name, rows in (('detections.csv', det_rows), ('timing.csv', time_rows)):
        if rows:
            with open(os.path.join(config.dataset.test_output, name), 'w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    for k in ('Pre (ms)', 'Model (ms)', 'Post (ms)', 'Total (ms)'):
        v = np.array([t[k] for t in time_rows])
        logger.info(f'{k:<11} avg {v.mean():7.2f}  P95 {np.percentile(v, 95):7.2f}')
    logger.info(f'{len(det_rows)} detections in {len(time_rows)} images, saved in {config.dataset.test_output}')
    return det_rows, time_rows


if __name__ == '__main__':
    # Load configuration
    config = OmegaConf.load('Config/main_config.yaml')
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(config.environment.cuda_launch_blocking)
    if not config.get('model_name'):  # default: the training run folder that contains weights/best.onnx
        config.model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(config.onnx_infer.weights))))

    # Generate timestamp and run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = os.path.join('bin', f'Run_{timestamp}_{config.model_name}_onnx_infer')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Update config paths
    config.dataset.save_path = checkpoints_dir
    config.dataset.test_output = checkpoints_dir

    # Save the config used for this run
    OmegaConf.save(config, os.path.join(checkpoints_dir, "config_used.yaml"))

    prepare_logger(checkpoints_dir, 'onnx_infer.log')
    logger.info(f'ONNX inference output directory: {checkpoints_dir}')
    main(config)
