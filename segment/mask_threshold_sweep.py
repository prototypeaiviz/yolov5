"""Run the full model report once per mask threshold, for one or more models, to find the best pixel.mask_thres.

A pixel belongs to a predicted mask when its mask probability is above the threshold (YOLOv5 default 0.5).
A higher threshold gives smaller masks: precision goes up, recall goes down. The threshold is post-processing
only, so the model and its speed do not change.

Settings come from Config/main_config.yaml: model, dataset, map and pixel as for segment/model_report.py,
plus the mask_sweep section:
    models      list of best.pt paths (empty = model.weights)
    data_yaml   dataset yaml for the sweep (empty = dataset.data_yaml)
    thresholds  mask thresholds to test
    plots       false = no prediction plots (faster)
Run from the repo root:
    python segment/mask_threshold_sweep.py

Outputs in bin/Run_<timestamp>_mask_sweep/:
    config_used.yaml, sweep.log
    <model>/thres_<value>/  full report for that model and threshold (report.json, summary.md, sheet.csv, ...)
    mask_sweep.csv / .tsv   one row per model and threshold: Model, Mask thres + the results-sheet columns
                            (.tsv pastes into Excel as separate cells)
"""
from datetime import datetime
import os
import sys

# Never let YOLOv5 pip-install missing packages into the shared environment. Must be set before utils.general loads.
os.environ.setdefault('YOLOv5_AUTOINSTALL', 'False')

from loguru import logger
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from segment import model_report  # noqa: E402

LABEL = 'Torch'


def model_name_of(weights):
    """Training run folder name, e.g. .../v18_yolov5s_seg_640/weights/best.pt -> v18_yolov5s_seg_640."""
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(weights))))


def log_table(rows):
    logger.info(f"{'model':<34} {'thres':>6} {'IoU':>7} {'Prec':>7} {'Recall':>7} {'mAP50-95':>9} {'mAP75':>7} "
                f"{'Post ms':>8}")
    for r in rows:
        logger.info(f"{r['Model']:<34} {r['Mask thres']:>6.2f} {r[f'IoU {LABEL}']:>7.4f} "
                    f"{r[f'Precision {LABEL}']:>7.4f} {r[f'Recall {LABEL}']:>7.4f} "
                    f"{r[f'Mask mAP50-95 {LABEL}']:>9.4f} {r[f'Mask mAP75 {LABEL}']:>7.4f} "
                    f"{r[f'{LABEL} Post (ms)']:>8.2f}")


def main(config):
    sw = config.mask_sweep
    models = list(sw.get('models') or [config.model.weights])
    data_yaml = sw.get('data_yaml') or config.dataset.data_yaml
    for w in models + [data_yaml]:
        if not os.path.exists(w):
            raise SystemExit(f'Not found: {w}')

    rows = []
    for weights in models:
        name = model_name_of(weights)
        for t in sw.thresholds:
            t = float(t)
            cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            cfg.model_name = name
            cfg.model.weights = weights
            cfg.dataset.data_yaml = data_yaml
            cfg.pixel.mask_thres = t
            cfg.dataset.test_output = os.path.join(config.dataset.save_path, name, f'thres_{t:.2f}')
            cfg.report.summary_csv = ''  # the sweep writes its own table
            if not sw.plots:
                cfg.plots.num_images = 0
                cfg.plots.num_worst = 0
            os.makedirs(cfg.dataset.test_output, exist_ok=True)

            logger.info(f'===== {name}: mask threshold {t:.2f} =====')
            result = model_report.main(cfg)
            rows.append({'Model': name, 'Mask thres': t, **result['sheet']})
            for fname, delim in (('mask_sweep.csv', ','), ('mask_sweep.tsv', '\t')):  # updated after each report
                model_report.write_csv(os.path.join(config.dataset.save_path, fname), rows, delimiter=delim)

    logger.info('===== Mask threshold sweep =====')
    log_table(rows)
    for name in dict.fromkeys(r['Model'] for r in rows):
        mine = [r for r in rows if r['Model'] == name]
        best_iou = max(mine, key=lambda r: r[f'IoU {LABEL}'])
        best_map = max(mine, key=lambda r: r[f'Mask mAP50-95 {LABEL}'])
        logger.info(f"{name}: best by IoU mask_thres {best_iou['Mask thres']:.2f} "
                    f"(IoU {best_iou[f'IoU {LABEL}']:.4f}), best by Mask mAP50-95 mask_thres "
                    f"{best_map['Mask thres']:.2f} (mAP50-95 {best_map[f'Mask mAP50-95 {LABEL}']:.4f})")
    logger.info(f"Table: {os.path.join(config.dataset.save_path, 'mask_sweep.tsv')}")
    return rows


if __name__ == '__main__':
    # Load configuration
    config = OmegaConf.load('Config/main_config.yaml')
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(config.environment.cuda_launch_blocking)

    # Generate timestamp and run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = os.path.join('bin', f'Run_{timestamp}_mask_sweep')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Update config paths
    config.dataset.save_path = checkpoints_dir
    config.dataset.test_output = checkpoints_dir

    # Save the config used for this run
    OmegaConf.save(config, os.path.join(checkpoints_dir, "config_used.yaml"))

    model_report.prepare_logger(checkpoints_dir, 'sweep.log')
    logger.info(f'Mask threshold sweep output directory: {checkpoints_dir}')
    main(config)
