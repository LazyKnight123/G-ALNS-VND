# -*- coding: utf-8 -*-
"""Shared helpers for the Zunhua CLRP experiments.

Responsibilities: load the main solver, freeze each run's config, isolate outputs, and write a traceable run table.
All comparisons are paired on the same random seeds; statistics and extra plotting live in a separate script.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent  # repository parent: GIS data only
ROOT = SCRIPTS_DIR  # script workspace: local outputs
MAIN_SCRIPT = SCRIPTS_DIR / '02_solve_zunhua_CLRP_ALNS.py'
MODULE_NAME = 'clrp_zunhua_main'
BASE_MATRIX_VERSION = '20260731_zunhua_scalable_detour_v1_scripts'
ABLATION_KEYS = ('CLRP_ABLATION_IGNORE_DEM_LOS', 'CLRP_ABLATION_PLANAR_DISTANCE',
                  'CLRP_ABLATION_DISABLE_FACILITY_OPS',
                  'CLRP_ABLATION_DISABLE_ALNS_FACILITY_OPS')


def load_solver(path: str | Path | None = None) -> Any:
    """Load the main script under a stable module name so Windows spawn workers can import it."""
    source = Path(path).expanduser().resolve() if path else MAIN_SCRIPT
    if not source.is_file():
        raise FileNotFoundError(f'main solver script not found: {source}')
    if str(source.parent) not in sys.path:
        sys.path.insert(0, str(source.parent))
    cached = sys.modules.get(MODULE_NAME)
    if cached and Path(getattr(cached, '__file__', '')).resolve() == source:
        return cached
    spec = importlib.util.spec_from_file_location(MODULE_NAME, source)
    if not spec or not spec.loader:
        raise ImportError(f'cannot load: {source}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    # The same Python process may run several cases in sequence; keep the original config so one case cannot leak into the next.
    module._EXPERIMENT_BASELINE_CONFIG = config_snapshot(module)
    return module


def _sync_env(module: Any) -> None:
    for key in ABLATION_KEYS:
        os.environ[key] = '1' if bool(getattr(module, key, False)) else '0'


def worker_init(module_path: str, snapshot: Mapping[str, Any]) -> None:
    """Restore this run's config after ProcessPoolExecutor spawn in the main script."""
    module = load_solver(module_path)
    for key, value in snapshot.items():
        setattr(module, key, value)
    _sync_env(module)


def config_snapshot(module: Any) -> dict[str, Any]:
    simple = (type(None), bool, int, float, str, bytes)
    return {key: value for key, value in vars(module).items()
            if (key.startswith('CLRP_') or key == '_OUTPUT_DIR')
            and isinstance(value, simple)}


# Legacy interface used on the Windows spawn path; keep it so the main experiment script does not need changes.
def collect_module_config_snapshot(module: Any) -> dict[str, Any]:
    return config_snapshot(module)


def multiprocess_worker_init(module_path: str, module_name: str, snapshot: Mapping[str, Any]) -> None:
    del module_name  # This helper always uses a fixed module name so parent and child processes stay consistent.
    worker_init(module_path, snapshot)


def apply_config(module: Any, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Restore baseline switches, apply overrides, and fill derived parameters."""
    for key, value in getattr(module, '_EXPERIMENT_BASELINE_CONFIG', {}).items():
        setattr(module, key, value)
    baseline = {
        'CLRP_ENABLE_NO_FLY_ZONE': True,
        'CLRP_ABLATION_IGNORE_DEM_LOS': False,
        'CLRP_ABLATION_PLANAR_DISTANCE': False,
        'CLRP_ABLATION_DISABLE_FACILITY_OPS': False,
        'CLRP_ABLATION_DISABLE_ALNS_FACILITY_OPS': False,
        'CLRP_INITIAL_FACILITY_REFINEMENT': True,
    }
    merged = {**baseline, **overrides}
    for key, value in merged.items():
        if not hasattr(module, key):
            raise AttributeError(f'main script has no config attribute: {key}')
        setattr(module, key, value)
    if 'CLRP_R2_UAV_SERVICE_RADIUS' in merged:
        module.CLRP_MAX_ROUTE_LENGTH = 2.0 * float(module.CLRP_R2_UAV_SERVICE_RADIUS)
        merged['CLRP_MAX_ROUTE_LENGTH'] = module.CLRP_MAX_ROUTE_LENGTH
    module.CLRP_EXPERIMENT_NO_FLY_SHP = getattr(module, '_RAILWAY_SHP', '') if module.CLRP_ENABLE_NO_FLY_ZONE else ''
    merged['CLRP_EXPERIMENT_NO_FLY_SHP'] = module.CLRP_EXPERIMENT_NO_FLY_SHP
    _sync_env(module)
    return merged


def tag_value(value: Any) -> str:
    return f'{float(value):g}'.replace('.', 'p') if isinstance(value, float) else str(value)


def run_key(experiment: Any, case_id: Any, seed: Any) -> tuple[str, str, int]:
    """Resume key at the experiment-book level: (experiment, case_id, seed)."""
    return (str(experiment), str(case_id), int(seed))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write CSV atomically so an interrupt cannot truncate an existing results table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ['run_id']
    tmp_path = path.with_name(path.name + '.tmp')
    with tmp_path.open('w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)
    return path


def _max_iter_of(row: Mapping[str, Any]) -> int | None:
    raw = row.get('max_iter', '')
    if raw is None or str(raw).strip() == '':
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def is_successful_run(row: Mapping[str, Any], *, max_iter: int | None = None) -> bool:
    """status=success; if max_iter is given, the stored value must also match before the run is skipped."""
    if str(row.get('status', '')).strip().lower() != 'success':
        return False
    if max_iter is None:
        return True
    stored = _max_iter_of(row)
    return stored is None or stored == int(max_iter)


def index_runs(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Keep the last row for each key (later CSV rows overwrite earlier ones)."""
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        try:
            key = run_key(row['experiment'], row['case_id'], row['seed'])
        except (KeyError, TypeError, ValueError):
            continue
        indexed[key] = dict(row)
    return indexed


def seeds_to_run(existing: Sequence[Mapping[str, Any]], *, experiment: str, case_id: str,
                 seeds: Sequence[int], max_iter: int, resume: bool, retry_failed: bool) -> list[int]:
    """Decide which seeds still need solving from existing rows. If resume=False, rerun all seeds."""
    if not resume:
        return [int(x) for x in seeds]
    indexed = index_runs(existing)
    pending: list[int] = []
    for seed in seeds:
        seed = int(seed)
        row = indexed.get(run_key(experiment, case_id, seed))
        if row is None:
            pending.append(seed)
            continue
        if is_successful_run(row, max_iter=max_iter):
            continue
        if retry_failed:
            pending.append(seed)
    return pending


def merge_run_records(existing: Sequence[Mapping[str, Any]],
                      new_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge by (experiment, case_id, seed); new rows overwrite old ones and the rest are kept."""
    merged = index_runs(existing)
    for row in new_rows:
        merged[run_key(row['experiment'], row['case_id'], row['seed'])] = dict(row)
    return sorted(merged.values(), key=lambda r: (str(r.get('experiment', '')), str(r.get('case_id', '')),
                                                  int(r.get('seed', 0)), str(r.get('started_at', ''))))


def run_case(module: Any, *, case_id: str, experiment: str, seeds: Sequence[int], max_iter: int,
             output_dir: Path, overrides: Mapping[str, Any], max_workers: int | None,
             geometry_changed: bool, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Run one case and return one long-table row per seed; solver errors are not swallowed."""
    if not seeds:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    effective = dict(overrides)
    effective.update({'CLRP_ALNS_RESUME': False, 'CLRP_REPLOT_RESULT_JSON': '',
                      'CLRP_MAX_ITER': int(max_iter), '_OUTPUT_DIR': str(output_dir),
                      'CLRP_MATRIX_CHECKPOINT_VERSION':
                      f'{BASE_MATRIX_VERSION}_{case_id}' if geometry_changed else BASE_MATRIX_VERSION})
    applied = apply_config(module, effective)
    workers = int(max_workers or getattr(module, 'CLRP_PARALLEL_MAX_WORKERS', 8))
    started = datetime.now().isoformat(timespec='seconds')
    summaries = module.run_parallel_experiments(random_seeds=[int(x) for x in seeds], max_iter=int(max_iter),
        no_fly_shp=module.CLRP_EXPERIMENT_NO_FLY_SHP, max_workers=workers, output_root=str(output_dir))
    rows = []
    for item in summaries:
        rows.append({'run_id': f'{experiment}:{case_id}:seed_{item.get("seed", "")}',
            'experiment': experiment, 'case_id': case_id, 'seed': item.get('seed', ''),
            'status': item.get('status', ''), 'max_iter': max_iter, 'started_at': started,
            'output_dir': str(output_dir), 'best_total_cost': item.get('best_total_cost', ''),
            'build_cost': item.get('build_cost', ''), 'transport_cost': item.get('transport_cost', ''),
            'n_open_depots': item.get('n_open_depots', ''), 'n_routes': item.get('n_routes', ''),
            'solution_result_json': item.get('solution_result_json', ''),
            'solution_png': item.get('solution_png', ''), **metadata,
            'config_json': json.dumps(applied, ensure_ascii=False, sort_keys=True)})
    return rows
