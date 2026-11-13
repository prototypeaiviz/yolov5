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
    if 'proto2x-nm64' in cfg_name:
        return f'stride-2 prototypes, transposed conv 2x2, 64 prototypes ({imgsz // 2}x{imgsz // 2})'
    if 'proto2x-rconv' in cfg_name:
        return f'stride-2 prototypes, bilinear upsample + 3x3 conv ({imgsz // 2}x{imgsz // 2})'
    if 'proto2x-dw' in cfg_name:
        return f'stride-2 prototypes, bilinear upsample + depthwise 3x3 ({imgsz // 2}x{imgsz // 2})'
    if 'proto2x' in cfg_name:
        return f'stride-2 prototypes, transposed conv 2x2 ({imgsz // 2}x{imgsz // 2})'
    return f'standard prototypes ({imgsz // 4}x{imgsz // 4})'


EPOCHS_DONE = re.compile(r'(\d+) epochs completed in ([0-9.]+) hours')


def train_log_hours(name, train_logs):
    """Hours printed by train.py at the end of training ('N epochs completed in H hours') in the runner's
    training log <name>_bs<batch>.log; None if no finished log is found."""
    hours = None
    for log in train_logs.get(name, []):
        with open(log, errors='ignore') as f:
            m = EPOCHS_DONE.findall(f.read())
        if m:
            hours = float(m[-1][1])
    return hours


def events_hours(run_dir):
    """Training duration from the TensorBoard events file (first to last logged event, one per epoch); None if
    the file or the tensorboard package is missing."""
    events = glob.glob(os.path.join(run_dir, 'events.out.tfevents.*'))
    if not events:
        return None
    try:
        from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
        times = [e.wall_time for e in EventFileLoader(events[0]).Load()]
    except Exception as e:  # missing package or unreadable file
        logger.warning(f'Cannot read {events[0]}: {e!r}')
        return None
    return (max(times) - min(times)) / 3600 if len(times) > 1 else None


def train_time(run_dir, runner_rows, train_logs=None):
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
    log_hours = train_log_hours(name, train_logs or {})
    if log_hours is not None:  # exact: train.py's own measurement of the successful attempt
        return start, end, round(log_hours, 2), 'train.py log (epochs completed in)', note, host or '', ''
    ev_hours = events_hours(run_dir)
    if ev_hours is not None:  # exact: timestamps logged every epoch by the training itself
        return start, end, round(ev_hours, 2), 'TensorBoard events file (logged times)', note, host or '', ''
    if name in runner_rows and runner_rows[name].get('Train minutes'):
        hours = float(runner_rows[name]['Train minutes']) / 60  # includes out-of-memory retries
        return start, end, round(hours, 2), 'experiments.csv (incl. retries)', note, host or '', ''
    if start and end and end > start:
        return start, end, round((end - start) / 3600, 2), 'events file -> last.pt time', note, host or '', ''
    return start, end, None, 'unknown', note, host or '', ''


def read_train_run(run_dir, runner_rows, train_logs=None):
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
    start, end, hours, time_source, note, host, gpu = train_time(run_dir, runner_rows, train_logs)
    mosaic = float(hyp.get('mosaic', 1.0))
    d = {'Run name': name, 'Backbone/Config': cfg_name, 'Train Resolution': imgsz, 'Batch Size': opt['batch_size'],
         'Epochs': opt['epochs'], 'Epochs done': epochs_done, 'Best epoch (val mask mAP)': best_epoch,
         'Mosaic': 'no mosaic' if mosaic == 0 else f'mosaic {mosaic:g}',
         'Mask prototypes': proto_description(cfg_name, imgsz), 'Mask ratio': opt.get('mask_ratio', 4),
         'mask_max': hyp.get('mask_max', 0), 'Overlap masks': not opt.get('no_overlap', False),
         'Segmentation loss multiplier': hyp.get('seg_gain', 1.0), 'Mask Dice weight': hyp.get('dice_gain', 0.0),
         'Rect': opt.get('rect', False), 'Optimizer': opt.get('optimizer', ''), 'Pretrained': opt['weights'],
         'Data yaml': opt.get('data', ''), 'Start Date': date_str(start), 'End Date': date_str(end),
         'Time to Train (h)': hours, 'Train time source': time_source, 'Train machine': host, 'Train GPU': gpu,
         'Train note': note}
    parts = [cfg_name, d['Mask prototypes'], f'{imgsz}x{imgsz}', f'{opt["epochs"]} epochs (best {best_epoch})',
             f'batch {opt["batch_size"]}' + (f' ({note})' if note else ''), d['Mosaic'],
             f'mask-ratio {d["Mask ratio"]}'] + ([f'mask_max {d["mask_max"]}'] if d['mask_max'] else []) + \
        (['--rect'] if d['Rect'] else []) + ([] if d['Overlap masks'] else ['--no-overlap']) + \
        [f'seg_gain {d["Segmentation loss multiplier"]:g}'] + \
        ([f'dice_gain {d["Mask Dice weight"]:g}'] if d['Mask Dice weight'] else [])
    d['Configuration'] = ', '.join(parts)
    return d


def read_report(rep_dir):
    """(run name, timestamp, fields) of a model_report.py output folder, or None."""
    path = os.path.join(rep_dir, 'test', 'report.json')
    if not os.path.exists(path):  # mask_threshold_sweep.py writes <sweep>/<model>/thres_<t>/report.json
        path = os.path.join(rep_dir, 'report.json')
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
              'Torch conf (P/R/IoU)': s.get('pixel_conf_thres', ''), 'Torch GPU': s.get('device', ''),
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

    train_logs = {}  # run name -> training logs <name>_bs<batch>.log written by the runner
    for d in expand(rc.runner_logs):
        for log in sorted(glob.glob(os.path.join(d, 'train', '*_bs*.log'))):
            train_logs.setdefault(re.sub(r'_bs\d+\.log$', '', os.path.basename(log)), []).append(log)

    runs = {}
    for d in expand(rc.train_runs):
        if os.path.exists(os.path.join(d, 'opt.yaml')):
            runs[os.path.basename(d)] = read_train_run(d, runner_rows, train_logs)
    logger.info(f'{len(runs)} training runs')

    reports = {}
    for d in expand(rc.report_runs):
        r = read_report(d)
        if r and (r[0] not in reports or r[1] > reports[r[0]][0]):  # newest report of each run
            reports[r[0]] = (r[1], r[2])
    logger.info(f'{len(reports)} Torch reports')

    # TensorRT rows: newest per (run, machine)
    trt, trt_all = {}, []
    for d in expand(rc.trt_runs):
        r = read_trt(d)
        if r:
            run, row = r
            trt_all.append(row)
            key = (run, row['Machine (host)'])
            if key not in trt or row['Run date'] > trt[key]['Run date']:
                trt[key] = row
    hosts = sorted({h for _, h in trt})
    logger.info(f'{len(trt_all)} TensorRT result rows from machines {hosts}')
    names = dict(rc.get('machine_names') or {})
    trt_host = rc.get('trt_machine') or (hosts[0] if hosts else '')
    final_host = rc.get('final_machine') or ''
    label = lambda h, t: f"{names.get(h, h)} / {t['OS']} / {t['GPU']}"  # noqa: E731

    defaults = OmegaConf.to_container(rc.defaults, resolve=True)
    sheet_rows, detail_rows = [], []
    excluded = set(rc.get('exclude') or [])
    all_runs = sorted((set(runs) | set(reports) | {r for r, _ in trt}) - excluded)
    for name in all_runs:
        row = dict(defaults)
        tr = runs.get(name)
        if tr:
            row.update(tr)
            row['Inspection Resolution'] = tr['Train Resolution']
        else:  # training folder not available: training time from the runner logs only
            row['Run name'] = name
            hours = train_log_hours(name, train_logs)
            if hours is not None:
                row.update({'Time to Train (h)': round(hours, 2), 'Train time source': 'train.py log (epochs completed in)'})
            elif runner_rows.get(name, {}).get('Train minutes'):
                row.update({'Time to Train (h)': round(float(runner_rows[name]['Train minutes']) / 60, 2),
                            'Train time source': 'experiments.csv (incl. retries)'})
            logger.warning(f'{name}: no training folder (opt.yaml) found')
        if name in reports:
            row.update(reports[name][1])
        else:
            logger.warning(f'{name}: no Torch report found')
        notes = [n for n in [(tr or {}).get('Train note', ''),
                             f"Torch metrics at mask threshold {row.get('Torch mask threshold')}"
                             + (f", Torch on {row.get('Torch GPU')}" if row.get('Torch GPU') else '')
                             if name in reports else 'no Torch report'] if n]
        t = trt.get((name, trt_host))
        if t:  # TRT columns = the TensorRT test machine
            row.update({k: t.get(k) for k in TRT_KEYS})
            row['Machine'] = label(trt_host, t)
            row['Inspection Resolution'] = t['Inspection Resolution']
            row['TRT Precision Mode'] = t['TRT Precision Mode']
            row.update({'TRT mask threshold': t.get('Mask threshold'), 'TRT mask upsample': t.get('Mask upsample'),
                        'TRT post-processing': t.get('Post-processing on'), 'TRT build': t.get('Build type'),
                        'TRT steady-state model (ms)': t.get('TRT Model steady-state median (ms)'),
                        'TRT run date': t.get('Run date'), 'TRT report': t.get('Output folder')})
            notes.append(f"TRT mask threshold {t.get('Mask threshold')} {t.get('Mask upsample')}, "
                         f"post-processing {t.get('Post-processing on')}")
        f = trt.get((name, final_host)) if final_host else None
        if f:  # Final Machine columns = the production machine
            row.update({'Final Machine Avg (ms)': f.get('TRT Total Avg (ms)'), 'Final Machine P95 (ms)': f.get('TRT P95 (ms)'),
                        'Final Machine': label(final_host, f), 'Final Machine Pre (ms)': f.get('TRT Pre (ms)'),
                        'Final Machine Model (ms)': f.get('TRT Model (ms)'), 'Final Machine Post (ms)': f.get('TRT Post (ms)'),
                        'Final Machine steady-state model (ms)': f.get('TRT Model steady-state median (ms)'),
                        'Final Machine Mask mAP50-95': f.get('Mask mAP50-95 TRT'), 'Final Machine IoU': f.get('IoU TRT'),
                        'Final Machine report': f.get('Output folder')})
            notes.append(f"Final Machine: {label(final_host, f)}")
        row['Notes'] = '; '.join(notes)
        sheet_rows.append({c: row.get(c) for c in SHEET_COLUMNS})
        detail_rows.append(row)

    out = config.dataset.save_path
    sheet = pd.DataFrame(sheet_rows, columns=SHEET_COLUMNS)
    details = pd.DataFrame(detail_rows)
    with pd.ExcelWriter(os.path.join(out, 'registry.xlsx')) as w:
        sheet.to_excel(w, sheet_name='Results sheet', index=False)
        details.to_excel(w, sheet_name='Details', index=False)
        pd.DataFrame(trt_all).to_excel(w, sheet_name='TRT runs', index=False)
        tcols = ['Run name', 'Backbone/Config', 'Train Resolution', 'Epochs', 'Batch Size', 'Start Date', 'End Date',
                 'Time to Train (h)', 'Train time source', 'Train machine', 'Train note']
        details.reindex(columns=tcols).to_excel(w, sheet_name='Training time', index=False)
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
