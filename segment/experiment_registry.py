"""Experiment registry: one results-sheet row per training run, with its configuration, training time, Torch report
and C++ TensorRT results.

Reads (all paths and glob patterns from the `registry` section of Config/main_config.yaml):
    train_runs   training run folders (train.py output): opt.yaml, results.csv, weights/, events file,
                 runner_info.json (written by segment/run_experiments.py since 2026-11-11)
    report_runs  segment/model_report.py output folders (Run_*_torch): test/report.json, config_used.yaml
    runner_logs  segment/run_experiments.py output folders (Run_*_experiments): experiments.csv (status, train minutes)
    trt_runs     C++ TensorRT evaluation folders (edu_cpp/TensorRT/bin/Run_*_eval): result_row.json
Matching: a report belongs to the training run named by the folder of its weights (.../<run>/weights/best.pt);
a TensorRT row by its "Model folder" (folder of the .onnx, an optional "ref_" prefix is ignored).

Writes bin/Run_<timestamp>_registry/:
    registry.xlsx   "Results sheet" (sheet column order, one row per training run x TensorRT machine),
                    "Details" (every collected field), "TRT runs" (all C++ result rows)
    registry.tsv    the "Results sheet" rows, to paste into Excel
    config_used.yaml, registry.log
Columns the code cannot know (Experiment ID, Final Machine ...) stay empty; constant ones come from registry.defaults.
"""
from datetime import datetime
import csv
import glob
import json
import os
import re

from loguru import logger
from omegaconf import OmegaConf
import pandas as pd
import yaml

SHEET_COLUMNS = [
    'Experiment ID', 'Machine', 'Start Date', 'End Date', 'Testing Data', 'Split', 'Model', 'Backbone/Config', 'Scope',
    'Inspection Resolution', 'Owner', 'Status', 'Run Type', 'Annotation_Type', 'Platform', 'Weights',
    'Dataset Version', 'Number of Classes', 'Train Resolution', 'Time to Train (h)', 'Is TRT Compatible?',
    'TRT Precision Mode', 'Batch Size',
    'Torch Pre (ms)', 'Torch Model (ms)', 'Torch Post (ms)', 'Torch Total Avg (ms)',
    'TRT Pre (ms)', 'TRT Model (ms)', 'TRT Post (ms)', 'TRT Total Avg (ms)', 'TRT P95 (ms)',
    'Final Machine Avg (ms)', 'Final Machine P95 (ms)',
    'Box mAP50 Torch', 'Box mAP50-95 Torch', 'Mask mAP50 Torch', 'Mask mAP50-95 Torch', 'Mask mAP75 Torch',
    'IoU Torch', 'Precision Torch', 'Recall Torch',
    'Box mAP50 TRT', 'Box mAP50-95 TRT', 'Mask mAP50 TRT', 'Mask mAP50-95 TRT', 'Mask mAP75 TRT',
    'IoU TRT', 'Precision TRT', 'Recall TRT', 'Product Mask mAP50-95 TRT',
    'Notes', 'Configuration']
TORCH_KEYS = ['Torch Pre (ms)', 'Torch Model (ms)', 'Torch Post (ms)', 'Torch Total Avg (ms)', 'Box mAP50 Torch',
              'Box mAP50-95 Torch', 'Mask mAP50 Torch', 'Mask mAP50-95 Torch', 'Mask mAP75 Torch', 'IoU Torch',
              'Precision Torch', 'Recall Torch']
TRT_KEYS = ['TRT Pre (ms)', 'TRT Model (ms)', 'TRT Post (ms)', 'TRT Total Avg (ms)', 'TRT P95 (ms)', 'Box mAP50 TRT',
            'Box mAP50-95 TRT', 'Mask mAP50 TRT', 'Mask mAP50-95 TRT', 'Mask mAP75 TRT', 'IoU TRT', 'Precision TRT',
            'Recall TRT', 'Product Mask mAP50-95 TRT']
RUN_STAMP = re.compile(r'Run_(\d{8}_\d{6})_')


def prepare_logger(log_dir, log_filename):
    """Configure loguru logger with file output"""
    logger.remove()
    logger.add(os.path.join(log_dir, log_filename),
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
    logger.add(lambda msg: print(msg, end=""))


def expand(patterns):
    """Folders matching the glob patterns, sorted, without duplicates."""
    out = []
    for p in patterns or []:
        out += [d for d in sorted(glob.glob(str(p))) if os.path.isdir(d) and d not in out]
    return out


def date_str(ts):
    return datetime.fromtimestamp(ts).strftime('%d-%m-%Y') if ts else ''


def proto_description(cfg_name, imgsz):
    """Mask prototype head of the model yaml: name and prototype size at this input size."""
    if 'proto2x-dw' in cfg_name:
        return f'stride-2 prototypes, bilinear upsample + depthwise 3x3 ({imgsz // 2}x{imgsz // 2})'
    if 'proto2x' in cfg_name:
        return f'stride-2 prototypes, transposed conv 2x2 ({imgsz // 2}x{imgsz // 2})'
    return f'standard prototypes ({imgsz // 4}x{imgsz // 4})'


def train_time(run_dir, runner_rows):
    """(start timestamp, end timestamp, hours, source, note) of the training run."""
    info_path = os.path.join(run_dir, 'runner_info.json')
    if os.path.exists(info_path):  # exact: written by run_experiments.py after the successful attempt
        info = json.load(open(info_path))
        start = datetime.fromisoformat(info['start']).timestamp()
        end = datetime.fromisoformat(info['end']).timestamp()
        return start, end, info['train_hours'], 'runner_info.json', info.get('note', ''), info.get('host', ''), info.get('gpu', '')
    name = os.path.basename(run_dir)
    events = glob.glob(os.path.join(run_dir, 'events.out.tfevents.*'))
    start = host = None
    if events:  # events.out.tfevents.<start unix time>.<host>.<pid>.<n>
        parts = os.path.basename(events[0]).split('.')
        start, host = float(parts[3]), '.'.join(parts[4:-2])
    last = [os.path.join(run_dir, 'weights', f) for f in ('last.pt', 'best.pt')]
    end = max((os.path.getmtime(f) for f in last if os.path.exists(f)), default=None)
    note = runner_rows.get(name, {}).get('Status', '')
    note = note.replace('trained', '').strip(' ()') if note.startswith('trained') else ''
    if name in runner_rows and runner_rows[name].get('Train minutes'):
        hours = float(runner_rows[name]['Train minutes']) / 60  # includes out-of-memory retries
        return start, end, round(hours, 2), 'experiments.csv (incl. retries)', note, host or '', ''
    if start and end and end > start:
        return start, end, round((end - start) / 3600, 2), 'events file -> last.pt time', note, host or '', ''
    return start, end, None, 'unknown', note, host or '', ''


def read_train_run(run_dir, runner_rows):
    opt = yaml.safe_load(open(os.path.join(run_dir, 'opt.yaml')))
    name = os.path.basename(run_dir)
    imgsz = int(opt['imgsz'])
    cfg_name = os.path.splitext(os.path.basename(opt.get('cfg') or ''))[0] or \
        os.path.splitext(os.path.basename(str(opt['weights'])))[0]
    hyp = opt.get('hyp') or {}
    epochs_done, best_epoch = None, None
    res = os.path.join(run_dir, 'results.csv')
    if os.path.exists(res):
        r = pd.read_csv(res)
        r.columns = [c.strip() for c in r.columns]
        r = r[r['metrics/mAP_0.5:0.95(M)'] > 0]
        epochs_done = len(r)
        best_epoch = int(r['epoch'][r['metrics/mAP_0.5:0.95(M)'].idxmax()]) if len(r) else None
    start, end, hours, time_source, note, host, gpu = train_time(run_dir, runner_rows)
    mosaic = float(hyp.get('mosaic', 1.0))
    d = {'Run name': name, 'Backbone/Config': cfg_name, 'Train Resolution': imgsz, 'Batch Size': opt['batch_size'],
         'Epochs': opt['epochs'], 'Epochs done': epochs_done, 'Best epoch (val mask mAP)': best_epoch,
         'Mosaic': 'no mosaic' if mosaic == 0 else f'mosaic {mosaic:g}',
         'Mask prototypes': proto_description(cfg_name, imgsz), 'Mask ratio': opt.get('mask_ratio', 4),
         'mask_max': hyp.get('mask_max', 0), 'Overlap masks': not opt.get('no_overlap', False),
         'Rect': opt.get('rect', False), 'Optimizer': opt.get('optimizer', ''), 'Pretrained': opt['weights'],
         'Data yaml': opt.get('data', ''), 'Start Date': date_str(start), 'End Date': date_str(end),
         'Time to Train (h)': hours, 'Train time source': time_source, 'Train machine': host, 'Train GPU': gpu,
         'Train note': note}
    parts = [cfg_name, d['Mask prototypes'], f'{imgsz}x{imgsz}', f'{opt["epochs"]} epochs (best {best_epoch})',
             f'batch {opt["batch_size"]}' + (f' ({note})' if note else ''), d['Mosaic'],
             f'mask-ratio {d["Mask ratio"]}'] + ([f'mask_max {d["mask_max"]}'] if d['mask_max'] else []) + \
        (['--rect'] if d['Rect'] else []) + ([] if d['Overlap masks'] else ['--no-overlap'])
    d['Configuration'] = ', '.join(parts)
    return d


def read_report(rep_dir):
    """(run name, timestamp, fields) of a model_report.py output folder, or None."""
    path = os.path.join(rep_dir, 'test', 'report.json')
    if not os.path.exists(path):
        return None
    rep = json.load(open(path))
    run = os.path.basename(os.path.dirname(os.path.dirname(rep['settings']['weights'])))
    m = RUN_STAMP.search(os.path.basename(rep_dir))
    d = {k: rep['report'].get(k) for k in TORCH_KEYS}
    s = rep['settings']
    d.update({'Torch report': rep_dir, 'Torch backend': s.get('backend', ''), 'Torch FP16': s.get('half', ''),
              'Torch mask threshold': s.get('mask_thres', '0.5 (default, older report)'),
              'Torch mask mode': s.get('mask_mode', ''),
              'Torch conf (P/R/IoU)': s.get('pixel_conf_thres', ''),
              'Torch instance mean IoU': rep.get('extra', {}).get('Instance mean IoU (matched)')})
    return run, (m.group(1) if m else ''), d


def read_trt(trt_dir):
    path = os.path.join(trt_dir, 'result_row.json')
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    run = r.get('Model folder', '')
    run = run[4:] if run.startswith('ref_') else run
    return run, r


def main(config):
    rc = config.registry
    runner_rows = {}
    for d in expand(rc.runner_logs):
        p = os.path.join(d, 'experiments.csv')
        if os.path.exists(p):
            with open(p) as f:
                for row in csv.DictReader(f):
                    if row.get('Status', '').startswith('trained'):
                        runner_rows[row['Experiment']] = row  # later logs win

    runs = {}
    for d in expand(rc.train_runs):
        if os.path.exists(os.path.join(d, 'opt.yaml')):
            runs[os.path.basename(d)] = read_train_run(d, runner_rows)
    logger.info(f'{len(runs)} training runs')

    reports = {}
    for d in expand(rc.report_runs):
        r = read_report(d)
        if r and (r[0] not in reports or r[1] > reports[r[0]][0]):  # newest report of each run
            reports[r[0]] = (r[1], r[2])
    logger.info(f'{len(reports)} Torch reports')

    trt = {}
    trt_all = []
    for d in expand(rc.trt_runs):
        r = read_trt(d)
        if r:
            trt.setdefault(r[0], []).append(r[1])
            trt_all.append(r[1])
    logger.info(f'{len(trt_all)} TensorRT result rows')

    defaults = OmegaConf.to_container(rc.defaults, resolve=True)
    sheet_rows, detail_rows = [], []
    for name, tr in sorted(runs.items()):
        base = dict(defaults)
        base.update({k: v for k, v in tr.items()})
        base['Inspection Resolution'] = tr['Train Resolution']
        if name in reports:
            base.update(reports[name][1])
        else:
            logger.warning(f'{name}: no Torch report found')
        notes = [n for n in [tr['Train note'], f"Torch metrics at mask threshold {base.get('Torch mask threshold')}"
                             if name in reports else 'no Torch report'] if n]
        for t in trt.get(name, [None]):  # one row per TensorRT machine / evaluation
            row = dict(base)
            row['Notes'] = '; '.join(notes)
            if t:
                row.update({k: t.get(k) for k in TRT_KEYS})
                row['Machine'] = f"{t['Machine (host)']} / {t['OS']} / {t['GPU']}"
                row['Inspection Resolution'] = t['Inspection Resolution']
                row['TRT Precision Mode'] = t['TRT Precision Mode']
                row.update({'TRT mask threshold': t.get('Mask threshold'), 'TRT mask upsample': t.get('Mask upsample'),
                            'TRT post-processing': t.get('Post-processing on'), 'TRT build': t.get('Build type'),
                            'TRT steady-state model (ms)': t.get('TRT Model steady-state median (ms)'),
                            'TRT run date': t.get('Run date'), 'TRT report': t.get('Output folder')})
                row['Notes'] += f"; TRT mask threshold {t.get('Mask threshold')} {t.get('Mask upsample')}, " \
                                f"post-processing {t.get('Post-processing on')}"
            else:
                logger.warning(f'{name}: no TensorRT result row found')
            sheet_rows.append({c: row.get(c) for c in SHEET_COLUMNS})
            detail_rows.append(row)

    out = config.dataset.save_path
    sheet = pd.DataFrame(sheet_rows, columns=SHEET_COLUMNS)
    details = pd.DataFrame(detail_rows)
    with pd.ExcelWriter(os.path.join(out, 'registry.xlsx')) as w:
        sheet.to_excel(w, sheet_name='Results sheet', index=False)
        details.to_excel(w, sheet_name='Details', index=False)
        pd.DataFrame(trt_all).to_excel(w, sheet_name='TRT runs', index=False)
    sheet.to_csv(os.path.join(out, 'registry.tsv'), sep='\t', index=False)
    logger.info(f'{len(sheet_rows)} rows -> {os.path.join(out, "registry.xlsx")} and registry.tsv')
    return sheet


if __name__ == '__main__':
    # Load configuration
    config = OmegaConf.load('Config/main_config.yaml')

    # Generate timestamp and run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints_dir = os.path.join('bin', f'Run_{timestamp}_registry')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Update config paths
    config.dataset.save_path = checkpoints_dir
    config.dataset.test_output = checkpoints_dir

    # Save the config used for this run
    OmegaConf.save(config, os.path.join(checkpoints_dir, "config_used.yaml"))

    prepare_logger(checkpoints_dir, 'registry.log')
    logger.info(f'Registry output directory: {checkpoints_dir}')
    main(config)
