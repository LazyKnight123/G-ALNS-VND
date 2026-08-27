# -*- coding: utf-8 -*-
"""Zunhua CLRP solver: DEM, line-of-sight, no-fly partitioning, and parallel ALNS (scalable detour edition).

Adapted from the DEM/LOS ALNS CLRP solver; geospatial inputs now come from FCLRP_Zunhua_data:
- Candidate sites: zunhua_random_depots_2000/zunhua_random_depots_2000.shp
- Study area: Zunhua administrative boundary (working CRS is metric zunhua_UTM50N.shp, projected from zunhua.shp)
- DEM: Zunhua_DEM_12.5_wgs84_study_area_interpolated.tif
- Farmland polygons used to retain panoramic points: 2020_class_11_12.shp
- Railway no-fly source: hebei_railway/hebei_railway.shp

Main pipeline:
load waypoints/depots -> DEM heights and line of sight -> block-wise 3D matrices -> visibility-greedy initial solution
-> initial depot-structure refinement (1→1/1→2/2→2/3→2/3→4) -> adaptive ALNS/VND -> JSON/CSV/NPZ persistence -> reproducible plots.

The number of open depots has no explicit bounds and there is no integer-program precheck for a minimum covering set;
it is optimized jointly by construction cost, route fixed cost, flight distance, and the depot-structure operators.
"""
import os
import sys
import csv
import json
import re
import glob
import hashlib
import pickle
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import combinations
import numpy as np
import rasterio
from lib_no_fly_zone_detour import build_navigation_distances, route_navigation_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import time
import math
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString, Point, shape as shapely_shape
from shapely.ops import unary_union, transform as shapely_transform
from shapely.prepared import prep
from shapely.strtree import STRtree
import shapefile
from pyproj import CRS, Transformer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)  # repository parent: GIS data only
_BASE_DIR = _SCRIPT_DIR  # script workspace: local outputs / checkpoints
_ZUNHUA_GIS_DIR = os.path.join(_REPO_ROOT, 'FCLRP_Zunhua_data')
# Read every .shp in the candidate-site directory; currently a random set of 2000 depots.
_STARTING_POINT_DIR = os.path.join(_ZUNHUA_GIS_DIR, 'zunhua_random_depots_2000')
# Geographic boundary source (WGS84); the solver must use the metric projected version.
_ADMIN_BOUNDARY_SHP_WGS84 = os.path.join(_ZUNHUA_GIS_DIR, 'zunhua_admin_boundary', 'zunhua.shp')
_ADMIN_BOUNDARY_SHP = os.path.join(_ZUNHUA_GIS_DIR, 'zunhua_admin_boundary', 'zunhua_UTM50N.shp')
CLRP_ENABLE_NO_FLY_ZONE = True

# Output naming: directory outputs/CLRP_{experiment_id}/; subfolders and files use the same experiment id.
_EXPERIMENT_ID = 'zunhua'
_OUTPUT_DIR = os.path.join(_BASE_DIR, 'outputs', f'CLRP_{_EXPERIMENT_ID}')
os.makedirs(_OUTPUT_DIR, exist_ok=True)
_SCENARIO_DIR_LABELS = {
    'with_no_fly_zone': 'with_no_fly_zone',
    'without_no_fly_zone': 'without_no_fly_zone',
}

def _output_stem(*parts):
    """Build a unified output stem: experiment_id_semantic_parts_..."""
    chunks = [_EXPERIMENT_ID]
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text:
            chunks.append(text)
    return '_'.join(chunks)

def _run_output_dirname(scenario_label, max_iter):
    """Unified run subdirectory, e.g. with_no_fly_zone_iter_1000"""
    scenario_dir = _SCENARIO_DIR_LABELS.get(str(scenario_label), _sanitize_filename(scenario_label))
    return f'{scenario_dir}_iter_{int(max_iter)}'
_RAILWAY_SHP = os.path.join(_ZUNHUA_GIS_DIR, 'hebei_railway', 'hebei_railway.shp')
_FARMLAND_SHP = os.path.join(_ZUNHUA_GIS_DIR, 'zunhua_farmland', '2020_class_11_12.shp')
_DEM_TIF = os.path.join(_ZUNHUA_GIS_DIR, 'zunhua_DEM', 'Zunhua_DEM_12.5_wgs84_study_area_interpolated.tif')
CLRP_RAILWAY_BUFFER_DISTANCE = 500.0
CLRP_NAVIGATION_CLEARANCE = 100.0
# Scalable detour-matrix parameters: keep near-tangent logic while limiting boundary nodes and access candidates.
CLRP_NAVIGATION_BOUNDARY_STEP = 50.0
CLRP_NAVIGATION_BOUNDARY_NEIGHBORS = 8
CLRP_NAVIGATION_BOUNDARY_LINK_NEIGHBORS = 2
CLRP_NAVIGATION_MARGIN = 1.0
CLRP_NAVIGATION_MAX_BOUNDARY_NODES = 6000
CLRP_NAVIGATION_TERMINAL_CANDIDATE_LIMIT = 64
CLRP_NAVIGATION_TARGET_CHUNK_SIZE = 256
_DEFAULT_NO_FLY_SHP = _RAILWAY_SHP if CLRP_ENABLE_NO_FLY_ZONE else ''
_LAST_UNPLANNABLE_WAYPOINTS = []
_LAST_PARTITION_UNPLANNABLE_WAYPOINTS = []
_LAST_UNPLANNABLE_DETAILS = []
_LAST_NODE_GROUND_ELEVATIONS = None
_LAST_NODE_FLIGHT_HEIGHTS = None
_LAST_DEPOT_WAYPOINT_VISIBILITY = None
_WAYPOINT_EXCLUSION_SHP = _RAILWAY_SHP
CLRP_WAYPOINT_CIRCLE_RADIUS = 700.0
CLRP_DEPOT_HEIGHT_AGL = 20.0
CLRP_WAYPOINT_HEIGHT_AGL = 300.0
CLRP_MAX_VISIBILITY_DISTANCE = 6000.0
CLRP_DEM_CLEARANCE = 20.0
CLRP_DEM_PROFILE_STEP = 30.0
CLRP_TERRAIN_INTERSECTION_TOLERANCE = 0.01
CLRP_REUSE_MATRIX_CHECKPOINTS = True
CLRP_MATRIX_CHECKPOINT_VERSION = '20260731_zunhua_scalable_detour_v1_scripts'
# Consistency tolerance for kilometre-scale float32 distance matrices. A strict 1e-6 m absolute tolerance is too tight,
# because the same components summed along different float paths can differ by millimetres.
CLRP_DISTANCE_MATRIX_CHECK_RTOL = 1e-6
CLRP_DISTANCE_MATRIX_CHECK_ATOL = 2e-3
_MATRIX_CHECKPOINT_DIR = os.path.join(_BASE_DIR, 'checkpoints', 'zunhua_scalable_detour_matrices')
os.makedirs(_MATRIX_CHECKPOINT_DIR, exist_ok=True)

def _compose_total_flight_distance(ascent, horizontal, descent):
    """Compose the total flight-distance matrix with a single float32 evaluation order.

    Always use ``(ascent + horizontal) + descent`` to avoid millimetre rounding differences between
    "sum in float64 then cast" and "cast each component then sum".
    When the ``CLRP_ABLATION_PLANAR_DISTANCE`` ablation is on, return horizontal distance only.
    """
    ascent = np.asarray(ascent, dtype=np.float32)
    horizontal = np.asarray(horizontal, dtype=np.float32)
    descent = np.asarray(descent, dtype=np.float32)
    if ascent.shape != horizontal.shape or ascent.shape != descent.shape:
        raise ValueError('ascent, horizontal, and descent distance matrices have inconsistent shapes')
    if _ablation_planar_distance():
        return np.asarray(horizontal, dtype=np.float32).copy()
    with np.errstate(invalid='ignore', over='ignore'):
        return np.add(
            np.add(ascent, horizontal, dtype=np.float32),
            descent,
            dtype=np.float32,
        )

def _distance_matrix_consistency_report(supplied_total, rebuilt_total):
    """Return a consistency report between the supplied total matrix and the rebuilt component sum."""
    supplied = np.asarray(supplied_total, dtype=np.float32)
    rebuilt = np.asarray(rebuilt_total, dtype=np.float32)
    if supplied.shape != rebuilt.shape:
        return {
            'consistent': False,
            'shape_mismatch': True,
            'supplied_shape': tuple(supplied.shape),
            'rebuilt_shape': tuple(rebuilt.shape),
        }

    finite_supplied = np.isfinite(supplied)
    finite_rebuilt = np.isfinite(rebuilt)
    finite_pattern_mismatch = int(np.count_nonzero(finite_supplied != finite_rebuilt))
    finite = finite_supplied & finite_rebuilt
    if np.any(finite):
        supplied_finite = supplied[finite].astype(np.float64)
        rebuilt_finite = rebuilt[finite].astype(np.float64)
        differences = np.abs(supplied_finite - rebuilt_finite)
        close = np.isclose(
            supplied_finite,
            rebuilt_finite,
            rtol=CLRP_DISTANCE_MATRIX_CHECK_RTOL,
            atol=CLRP_DISTANCE_MATRIX_CHECK_ATOL,
        )
        mismatch_count = int(np.count_nonzero(~close))
        max_difference = float(np.max(differences))
        mean_difference = float(np.mean(differences))
    else:
        mismatch_count = 0
        max_difference = 0.0
        mean_difference = 0.0

    return {
        'consistent': finite_pattern_mismatch == 0 and mismatch_count == 0,
        'shape_mismatch': False,
        'finite_pattern_mismatch': finite_pattern_mismatch,
        'mismatch_count': mismatch_count,
        'max_difference': max_difference,
        'mean_difference': mean_difference,
    }

def _ensure_metric_admin_boundary():
    """If the metric study-area boundary is missing, project WGS84 zunhua.shp to EPSG:32650."""
    if os.path.isfile(_ADMIN_BOUNDARY_SHP) and os.path.isfile(os.path.splitext(_ADMIN_BOUNDARY_SHP)[0] + '.prj'):
        return _ADMIN_BOUNDARY_SHP
    if not os.path.isfile(_ADMIN_BOUNDARY_SHP_WGS84):
        raise FileNotFoundError(f'Zunhua boundary not found: {_ADMIN_BOUNDARY_SHP_WGS84}')
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError('geopandas is required to generate the metric study-area boundary') from exc
    gdf = gpd.read_file(_ADMIN_BOUNDARY_SHP_WGS84).to_crs('EPSG:32650')
    os.makedirs(os.path.dirname(_ADMIN_BOUNDARY_SHP), exist_ok=True)
    gdf.to_file(_ADMIN_BOUNDARY_SHP, encoding='utf-8')
    print(f'[Zunhua data] generated metric working boundary from WGS84: {_ADMIN_BOUNDARY_SHP}')
    return _ADMIN_BOUNDARY_SHP

_ensure_metric_admin_boundary()
# No upper bound on the number of open depots; this compatibility parameter is unused as a count constraint.
CLRP_N_SELECTED_DEPOTS = None
CLRP_CUSTOMER_DEMAND = 10.0
CLRP_R2_UAV_SERVICE_RADIUS = 10000.0
CLRP_DEPOT_CAPACITY = 1000.0
CLRP_DEPOT_BUILD_COST = 50000.0
CLRP_VEHICLE_FIXED_COST = 100.0
CLRP_DEPOT_EXTRA_VALUE = 0.0
CLRP_MAX_ROUTE_LENGTH = 2.0 * CLRP_R2_UAV_SERVICE_RADIUS
CLRP_RANDOM_SEED = 42
CLRP_MAX_ITER = 1000
CLRP_DESTROY_RATE = 0.1
CLRP_MIN_DESTROY_RATE = 0.05
CLRP_MAX_DESTROY_RATE = 0.2
CLRP_MAX_DESTROY_CUSTOMERS = 50
CLRP_T_START_WORSE_RATIO = 0.02
CLRP_T_START_ACCEPT_PROB = 0.6
CLRP_T_END_COST_RATIO = 1e-05
CLRP_LOCAL_SEARCH_INTERVAL = 50
CLRP_SEGMENT_LENGTH = 50
CLRP_STAGNATION_LIMIT = 220
CLRP_ELITE_SIZE = 10
CLRP_INITIAL_SWAP_PASSES = 8
# Initial depot-structure refinement: after visibility-greedy routing, apply 2→2, 3→2, and 3→4 local rebuilds.
CLRP_INITIAL_FACILITY_REFINEMENT = True
# Stop initial refinement after this many consecutive passes with no total-cost decrease.
CLRP_INITIAL_FACILITY_NO_IMPROVE_LIMIT = 20
# Hard cap so a sequence of improving moves cannot keep the initial stage running too long.
CLRP_INITIAL_FACILITY_MAX_PASSES = 80
# "Adjacent depots" are defined by a K-NN graph inside the same free-flight block.
CLRP_INITIAL_FACILITY_K_NEIGHBORS = 4
# Replacement depots must lie inside the buffered bounding box of the local waypoints.
CLRP_INITIAL_FACILITY_REGION_BUFFER = 1500.0
# Maximum candidate depots kept per local region, to control combinatorial size.
CLRP_INITIAL_FACILITY_CANDIDATE_POOL = 12
# Maximum depot combinations that receive a full route rebuild after visibility and proxy-cost screening.
CLRP_INITIAL_FACILITY_EXACT_TRIALS = 16
# Combined weight of waypoint count and the union area of 300 m coverage circles.
CLRP_INITIAL_FACILITY_COUNT_WEIGHT = 0.5
CLRP_INITIAL_FACILITY_AREA_WEIGHT = 0.5
# Accept a local depot rebuild only if cost falls by at least this amount.
CLRP_INITIAL_FACILITY_MIN_IMPROVEMENT = 1e-06
# In the initial stage, any move that adds an open depot (1→2, 3→4) must cut the block cost by more than 5%.
CLRP_INITIAL_FACILITY_EXPAND_MIN_REL_IMPROVEMENT = 0.05
# During ALNS, the three depot-structure operators (2→2, 3→2, 3→4) must cut cost by more than 1% to emit a candidate.
CLRP_ALNS_FACILITY_MIN_REL_IMPROVEMENT = 0.01
# Depot-structure operators are expensive, so their initial weights are lower than waypoint destroy operators; ALNS still adapts them later.
CLRP_ALNS_FACILITY_INITIAL_WEIGHT = 0.20
# Each ALNS depot-structure destroy picks one high-ranked local region at random and fully rebuilds it.
CLRP_ALNS_FACILITY_RANK_POOL = 20



CLRP_PARALLEL_RANDOM_SEEDS = [11, 23, 37, 72, 211]
CLRP_PARALLEL_MAX_WORKERS = 8
CLRP_CONSOLE_PROGRESS_INTERVAL = 1
# Lightweight ALNS resume: atomically save every 50 iterations and restore after a restart.
CLRP_ALNS_RESUME = True
CLRP_ALNS_CHECKPOINT_INTERVAL = 50
# Terminal VND has been removed from the main algorithm: route local search runs only inside ALNS iterations.
# A new global best requires at least this relative improvement (0.01% = 1e-4); smaller numerical jitter is ignored.
CLRP_BEST_MIN_REL_IMPROVEMENT = 1e-4
CLRP_EXPERIMENT_NO_FLY_SHP = _DEFAULT_NO_FLY_SHP
CLRP_REPLOT_RESULT_JSON = ''
# Ablation switches (off by default, baseline unchanged). Worker processes inherit the same environment variables.
CLRP_ABLATION_IGNORE_DEM_LOS = False
CLRP_ABLATION_PLANAR_DISTANCE = False
CLRP_ABLATION_DISABLE_FACILITY_OPS = False
# The switch below separates initial facility refinement from ALNS facility operators; it is off by default,
# so existing main experiments keep their original behaviour.
CLRP_ABLATION_DISABLE_ALNS_FACILITY_OPS = False
# Maximum in-iteration VND rounds. Set to 0 for a pure-ALNS ablation.
CLRP_VND_MAX_ROUNDS = 4
# Adaptive VND (off by default, preserving the previous fixed 4-round experiments): ordinary improving candidates get a
# light search; elite candidates near or better than the historic best get a deep search; periodic checks stay light.
CLRP_VND_ADAPTIVE_MODE = False
CLRP_VND_LIGHT_MAX_ROUNDS = 1
CLRP_VND_ELITE_MAX_ROUNDS = 4
CLRP_VND_ELITE_GAP_RATIO = 0.003
CLRP_VND_PERIODIC_INTERVAL = 100


def _env_flag_true(name):
    return str(os.environ.get(name, '')).strip().lower() in {'1', 'true', 'yes', 'on'}


def _ablation_ignore_dem_los():
    return bool(CLRP_ABLATION_IGNORE_DEM_LOS) or _env_flag_true('CLRP_ABLATION_IGNORE_DEM_LOS')


def _ablation_planar_distance():
    return bool(CLRP_ABLATION_PLANAR_DISTANCE) or _env_flag_true('CLRP_ABLATION_PLANAR_DISTANCE')


def _ablation_disable_facility_ops():
    return bool(CLRP_ABLATION_DISABLE_FACILITY_OPS) or _env_flag_true('CLRP_ABLATION_DISABLE_FACILITY_OPS')


def _ablation_disable_alns_facility_ops():
    """Keep the legacy master switch while also allowing ALNS facility operators to be disabled on their own."""
    return (_ablation_disable_facility_ops()
            or bool(CLRP_ABLATION_DISABLE_ALNS_FACILITY_OPS)
            or _env_flag_true('CLRP_ABLATION_DISABLE_ALNS_FACILITY_OPS'))


def _apply_clrp_random_seed(seed=CLRP_RANDOM_SEED):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed

def _dataset_signature_paths(path):
    if not path:
        return []
    path = os.path.abspath(os.fspath(path))
    root, ext = os.path.splitext(path)
    if ext.lower() == '.shp':
        candidates = [root + suffix for suffix in ('.shp', '.shx', '.dbf', '.prj', '.cpg')]
        return [candidate for candidate in candidates if os.path.exists(candidate)]
    return [path] if os.path.exists(path) else []

def _matrix_checkpoint_key(kind, arrays=(), config=None, source_paths=()):
    digest = hashlib.sha256()
    digest.update(CLRP_MATRIX_CHECKPOINT_VERSION.encode('utf-8'))
    digest.update(str(kind).encode('utf-8'))
    for path in source_paths:
        for dataset_file in _dataset_signature_paths(path):
            stat = os.stat(dataset_file)
            payload = (os.path.normcase(dataset_file), int(stat.st_size), int(stat.st_mtime_ns))
            digest.update(repr(payload).encode('utf-8'))
    if config:
        digest.update(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    for array in arrays:
        arr = np.ascontiguousarray(np.asarray(array))
        digest.update(str(arr.dtype).encode('ascii'))
        digest.update(repr(arr.shape).encode('ascii'))
        digest.update(arr.view(np.uint8).tobytes())
    return digest.hexdigest()[:24]

def _matrix_checkpoint_paths(kind, key):
    stem = os.path.join(_MATRIX_CHECKPOINT_DIR, f'{kind}_{key}')
    return (stem + '.npz', stem + '.json')

def _load_matrix_checkpoint(kind, key, expected_shapes=None):
    if not CLRP_REUSE_MATRIX_CHECKPOINTS:
        return None
    npz_path, json_path = _matrix_checkpoint_paths(kind, key)
    if not os.path.exists(npz_path) or not os.path.exists(json_path):
        return None
    try:
        with open(json_path, encoding='utf-8') as file_obj:
            metadata = json.load(file_obj)
        if metadata.get('version') != CLRP_MATRIX_CHECKPOINT_VERSION:
            return None
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        for name, shape in (expected_shapes or {}).items():
            if name not in arrays or tuple(arrays[name].shape) != tuple(shape):
                return None
        print(f'[matrix checkpoint] reused: {npz_path}')
        return (arrays, metadata)
    except Exception as exc:
        print(f'[matrix checkpoint] read failed, recomputing: {npz_path}; {exc}')
        return None

def _save_matrix_checkpoint(kind, key, arrays, metadata=None):
    npz_path, json_path = _matrix_checkpoint_paths(kind, key)
    payload = dict(metadata or {})
    payload.update({'version': CLRP_MATRIX_CHECKPOINT_VERSION, 'kind': str(kind), 'created_at': datetime.now().isoformat(timespec='seconds'), 'npz_path': npz_path})
    tmp_npz = npz_path + '.tmp'
    tmp_json = json_path + '.tmp'
    try:
        with open(tmp_npz, 'wb') as file_obj:
            np.savez_compressed(file_obj, **arrays)
        with open(tmp_json, 'w', encoding='utf-8') as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        os.replace(tmp_npz, npz_path)
        os.replace(tmp_json, json_path)
        print(f'[matrix checkpoint] saved: {npz_path}')
    finally:
        for temp_path in (tmp_npz, tmp_json):
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
    return npz_path

def _navigation_checkpoint_pickle_path(key):
    npz_path, _ = _matrix_checkpoint_paths('navigation', key)
    return os.path.splitext(npz_path)[0] + '.pkl'

def _save_navigation_checkpoint(key, horizontal_distances, no_fly_crossing, navigation_data, metadata=None):
    payload = dict(metadata or {})
    pickle_path = _navigation_checkpoint_pickle_path(key)
    pickle_saved = False
    if navigation_data is not None:
        tmp_pickle = pickle_path + '.tmp'
        try:
            with open(tmp_pickle, 'wb') as file_obj:
                pickle.dump(navigation_data, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_pickle, pickle_path)
            pickle_saved = True
        except Exception as exc:
            print(f'[navigation checkpoint] detour-path object is not serializable; saving matrices only: {exc}')
        finally:
            if os.path.exists(tmp_pickle):
                try:
                    os.remove(tmp_pickle)
                except OSError:
                    pass
    elif os.path.exists(pickle_path):
        try:
            os.remove(pickle_path)
        except OSError:
            pass
    payload.update({'navigation_data_required': navigation_data is not None, 'navigation_pickle_saved': pickle_saved, 'navigation_pickle_path': pickle_path if pickle_saved else None})
    return _save_matrix_checkpoint('navigation', key, arrays={'horizontal_navigation': np.asarray(horizontal_distances, dtype=np.float32), 'no_fly_crossing': np.asarray(no_fly_crossing, dtype=np.uint8)}, metadata=payload)

def _load_navigation_checkpoint(key, n_nodes):
    loaded = _load_matrix_checkpoint('navigation', key, expected_shapes={'horizontal_navigation': (n_nodes, n_nodes), 'no_fly_crossing': (n_nodes, n_nodes)})
    if loaded is None:
        return None
    arrays, metadata = loaded
    navigation_data = None
    if metadata.get('navigation_data_required'):
        pickle_path = metadata.get('navigation_pickle_path') or _navigation_checkpoint_pickle_path(key)
        if not metadata.get('navigation_pickle_saved') or not os.path.exists(pickle_path):
            print('[navigation checkpoint] missing detour-path object; recomputing 2D navigation.')
            return None
        try:
            with open(pickle_path, 'rb') as file_obj:
                navigation_data = pickle.load(file_obj)
        except Exception as exc:
            print(f'[navigation checkpoint] failed to read detour-path object; recomputing: {exc}')
            return None
    return (arrays['horizontal_navigation'].astype(np.float32, copy=False), arrays['no_fly_crossing'].astype(bool), navigation_data)

def _iter_polygonal_parts(geometry):
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from _iter_polygonal_parts(part)

def load_polygonal_area_geometry(shp_path):
    if not shp_path or not os.path.exists(shp_path):
        raise FileNotFoundError(f'Random candidate take-off/landing sites require an administrative polygon file, but it was not found: {shp_path}')
    sf = shapefile.Reader(shp_path)
    polygon_parts = []
    for shp_obj in sf.shapes():
        geom = shapely_shape(shp_obj.__geo_interface__)
        if not geom.is_valid:
            geom = geom.buffer(0)
        polygon_parts.extend(_iter_polygonal_parts(geom))
    if not polygon_parts:
        raise ValueError(f'administrative file contains no valid polygon geometry: {shp_path}')
    merged = unary_union(polygon_parts)
    if not merged.is_valid:
        merged = merged.buffer(0)
    if merged.is_empty or merged.area <= 0:
        raise ValueError(f'administrative polygon is empty or has invalid area: {shp_path}')
    return merged

def build_candidate_feasible_area(administrative_geometry, prohibited_polygons=None):
    valid_no_fly = [poly for poly in prohibited_polygons or [] if poly is not None and (not poly.is_empty)]
    feasible = administrative_geometry.difference(unary_union(valid_no_fly)) if valid_no_fly else administrative_geometry
    if not feasible.is_valid:
        feasible = feasible.buffer(0)
    polygon_parts = list(_iter_polygonal_parts(feasible))
    if not polygon_parts:
        raise ValueError('no usable area remains after subtracting exclusion polygons from the administrative boundary')
    feasible = unary_union(polygon_parts)
    print(f'feasible area for candidate sites: {feasible.area / 1000000.0:.2f} km²')
    return feasible

def _read_prj_crs(dataset_path):
    prj_path = os.path.splitext(os.fspath(dataset_path))[0] + '.prj'
    if not os.path.exists(prj_path):
        raise FileNotFoundError(f'vector dataset is missing a CRS file: {prj_path}')
    with open(prj_path, encoding='utf-8-sig') as file_obj:
        return CRS.from_wkt(file_obj.read())

def _require_metric_projected_crs(crs, label='working CRS'):
    crs = CRS.from_user_input(crs)
    if not crs.is_projected:
        raise ValueError(f'{label}must be a projected CRS; geographic lon/lat coordinates cannot be used directly.')
    axis_info = crs.axis_info or []
    if axis_info:
        factor = axis_info[0].unit_conversion_factor
        if factor is not None and abs(float(factor) - 1.0) > 1e-09:
            raise ValueError(f'{label} planar units must be metres, current unit={axis_info[0].unit_name}')
    return crs

def load_polygonal_geometry_reprojected(shp_path, target_crs):
    if not shp_path or not os.path.exists(shp_path):
        raise FileNotFoundError(f'polygon shapefile does not exist: {shp_path}')
    source_crs = _read_prj_crs(shp_path)
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    parts = []
    for shp_obj in shapefile.Reader(shp_path).shapes():
        geometry = shapely_shape(shp_obj.__geo_interface__)
        geometry = shapely_transform(transformer.transform, geometry)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        parts.extend(_iter_polygonal_parts(geometry))
    if not parts:
        raise ValueError(f'polygon shapefile contains no valid polygon features: {shp_path}')
    merged = unary_union(parts)
    if not merged.is_valid:
        merged = merged.buffer(0)
    return merged

class DEMTerrainModel:

    def __init__(self, dem_path, working_crs):
        if not dem_path or not os.path.exists(dem_path):
            raise FileNotFoundError(f'DEM file does not exist: {dem_path}')
        with rasterio.open(dem_path) as dataset:
            self.array = dataset.read(1).astype(float)
            self.transform = dataset.transform
            self.inverse_transform = ~dataset.transform
            self.dem_crs = CRS.from_user_input(dataset.crs)
            self.nodata = dataset.nodata
            self.height = dataset.height
            self.width = dataset.width
        self.working_crs = CRS.from_user_input(working_crs)
        self.to_dem = Transformer.from_crs(self.working_crs, self.dem_crs, always_xy=True)
        invalid = ~np.isfinite(self.array)
        if self.nodata is not None and np.isfinite(self.nodata):
            invalid |= np.isclose(self.array, float(self.nodata))
        self.invalid_mask = invalid

    def sample_many(self, xy):
        points = np.asarray(xy, dtype=float)
        if points.size == 0:
            return np.empty(0, dtype=float)
        x_dem, y_dem = self.to_dem.transform(points[:, 0], points[:, 1])
        col_f, row_f = self.inverse_transform * (np.asarray(x_dem), np.asarray(y_dem))
        cols = np.floor(col_f).astype(int)
        rows = np.floor(row_f).astype(int)
        valid = (rows >= 0) & (rows < self.height) & (cols >= 0) & (cols < self.width)
        values = np.full(len(points), np.nan, dtype=float)
        idx = np.flatnonzero(valid)
        if idx.size:
            sampled = self.array[rows[idx], cols[idx]]
            sampled_invalid = self.invalid_mask[rows[idx], cols[idx]]
            sampled = sampled.astype(float, copy=True)
            sampled[sampled_invalid] = np.nan
            values[idx] = sampled
        return values

    @staticmethod
    def _normalize_polyline(polyline):
        coords = np.asarray(polyline, dtype=float)
        if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) < 2:
            raise ValueError('a terrain profile needs at least two 2D coordinates')
        coords = coords[:, :2]
        keep = np.r_[True, np.any(np.abs(np.diff(coords, axis=0)) > 1e-10, axis=1)]
        coords = coords[keep]
        if len(coords) < 2:
            coords = np.vstack([coords[0], coords[0]])
        return coords

    def sample_profile(self, polyline, step=CLRP_DEM_PROFILE_STEP):
        coords = self._normalize_polyline(polyline)
        segments = np.diff(coords, axis=0)
        seg_lengths = np.linalg.norm(segments, axis=1)
        total = float(seg_lengths.sum())
        if total <= 1e-10:
            return (np.array([0.0]), self.sample_many(coords[:1]))
        sample_xy = []
        sample_d = []
        cumulative = 0.0
        step = max(float(step), 1.0)
        for seg_idx, (p0, p1, seg_len) in enumerate(zip(coords[:-1], coords[1:], seg_lengths)):
            if seg_len <= 1e-10:
                continue
            n_steps = max(1, int(math.ceil(seg_len / step)))
            fractions = np.linspace(0.0, 1.0, n_steps + 1)
            if seg_idx > 0:
                fractions = fractions[1:]
            for fraction in fractions:
                sample_xy.append(p0 + fraction * (p1 - p0))
                sample_d.append(cumulative + fraction * seg_len)
            cumulative += seg_len
        return (np.asarray(sample_d, dtype=float), self.sample_many(sample_xy))

    def line_of_sight(self, start_xy, end_xy, start_height, end_height, max_distance=CLRP_MAX_VISIBILITY_DISTANCE):
        planar = float(np.linalg.norm(np.asarray(end_xy) - np.asarray(start_xy)))
        distance_3d = math.hypot(planar, float(end_height) - float(start_height))
        if distance_3d >= float(max_distance) - 1e-10:
            return (False, 'OUT_OF_RANGE', distance_3d)
        # Ablation: ignore DEM terrain occlusion and keep only the line-of-sight range sphere.
        if _ablation_ignore_dem_los():
            return (True, 'VISIBLE_ABLATION_IGNORE_DEM', distance_3d)
        distances, terrain = self.sample_profile([start_xy, end_xy])
        valid = np.isfinite(terrain)
        if not np.any(valid):
            return (True, 'VISIBLE_NODATA_SKIPPED', distance_3d)
        total = max(planar, 1e-10)
        line_height = float(start_height) + (float(end_height) - float(start_height)) * np.clip(distances / total, 0.0, 1.0)
        blocked = terrain[valid] >= line_height[valid] - CLRP_TERRAIN_INTERSECTION_TOLERANCE
        if np.any(blocked):
            return (False, 'TERRAIN_BLOCKED', distance_3d)
        return (True, 'VISIBLE', distance_3d)

    def terrain_max(self, polyline):
        _, terrain = self.sample_profile(polyline)
        valid = terrain[np.isfinite(terrain)]
        return float(np.max(valid)) if valid.size else float('nan')

def _free_flight_blocks_geometry(administrative_boundary_shp, route_prohibited_polygons):
    boundary = load_polygonal_area_geometry(administrative_boundary_shp)
    forbidden = unary_union([p for p in route_prohibited_polygons or [] if p is not None and (not p.is_empty)])
    safe_forbidden = forbidden.buffer(CLRP_NAVIGATION_CLEARANCE) if not forbidden.is_empty else forbidden
    free = boundary.difference(safe_forbidden) if not safe_forbidden.is_empty else boundary
    return [p for p in _iter_polygonal_parts(free) if p.area > 1e-06]

def _point_block_ids(xy, blocks):
    ids = np.full(len(xy), -1, dtype=int)
    if not blocks:
        return ids
    tree = STRtree(blocks)
    for idx, point_xy in enumerate(np.asarray(xy, dtype=float)):
        point = Point(point_xy)
        candidates = tree.query(point, predicate='intersects')
        for block_idx in candidates:
            block_idx = int(block_idx)
            if blocks[block_idx].covers(point):
                ids[idx] = block_idx
                break
    return ids

def _navigation_pair_coordinates(navigation_data, coords, i, j):
    if navigation_data is None:
        return np.asarray([coords[i], coords[j]], dtype=float)
    routed = np.asarray(route_navigation_coordinates(navigation_data, [int(i), int(j)]), dtype=float)
    if routed.ndim != 2 or routed.shape[1] < 2 or len(routed) < 2:
        raise ValueError(f'detour coordinates for nodes {i}->{j} are invalid')
    return routed[:, :2]

def build_three_dimensional_flight_matrices(coords, flight_heights, dem_model, navigation_data, horizontal_distances, node_ids, global_coords):
    coords = np.asarray(coords, dtype=float)
    heights = np.asarray(flight_heights, dtype=float)
    node_ids = np.asarray(node_ids, dtype=int)
    global_coords = np.asarray(global_coords, dtype=float)
    n = len(coords)
    if len(node_ids) != n:
        raise ValueError('node_ids length must match the number of local nodes')
    total_matrix = np.zeros((n, n), dtype=np.float32)
    ascent_matrix = np.zeros((n, n), dtype=np.float32)
    horizontal_matrix = np.zeros((n, n), dtype=np.float32)
    descent_matrix = np.zeros((n, n), dtype=np.float32)
    cruise_matrix = np.full((n, n), np.nan, dtype=np.float32)
    np.fill_diagonal(cruise_matrix, heights)
    print(f'=== Precomputing DEM directional flight matrix ({n}×{n}) ===')
    for i in range(n):
        for j in range(i + 1, n):
            global_i, global_j = (int(node_ids[i]), int(node_ids[j]))
            horizontal = float(horizontal_distances[i][j])
            if np.isfinite(horizontal):
                try:
                    polyline = _navigation_pair_coordinates(navigation_data, global_coords, global_i, global_j)
                except ValueError:
                    print(f'[detour path not recoverable] nodes {global_i}->{global_j}; treat the leg as unreachable.')
                    horizontal = float('inf')
            if np.isfinite(horizontal):
                terrain_max = dem_model.terrain_max(polyline)
                terrain_required = -float('inf') if not np.isfinite(terrain_max) else terrain_max + CLRP_DEM_CLEARANCE
                cruise = max(float(heights[i]), float(heights[j]), terrain_required)
                ascent_ij = max(0.0, cruise - float(heights[i]))
                descent_ij = max(0.0, cruise - float(heights[j]))
                ascent_ji = max(0.0, cruise - float(heights[j]))
                descent_ji = max(0.0, cruise - float(heights[i]))
                total = ascent_ij + horizontal + descent_ij
            else:
                cruise = float('nan')
                ascent_ij = descent_ij = ascent_ji = descent_ji = float('inf')
                total = float('inf')
            total_matrix[i, j] = total_matrix[j, i] = total
            horizontal_matrix[i, j] = horizontal_matrix[j, i] = horizontal
            cruise_matrix[i, j] = cruise_matrix[j, i] = cruise
            ascent_matrix[i, j], descent_matrix[i, j] = (ascent_ij, descent_ij)
            ascent_matrix[j, i], descent_matrix[j, i] = (ascent_ji, descent_ji)
        if i and i % 50 == 0:
            print(f'  processed {i}/{n} nodes...')
    finite_edges = total_matrix[np.isfinite(total_matrix) & (total_matrix > 0)]
    if finite_edges.size:
        print(f'flight matrix done: mean={finite_edges.mean() / 1000:.3f} km, max={finite_edges.max() / 1000:.3f} km')
    return (total_matrix, ascent_matrix, horizontal_matrix, descent_matrix, cruise_matrix)

def load_candidate_depots_from_starting_points(administrative_boundary_shp, prohibited_polygons=None, start_point_dir=_STARTING_POINT_DIR):
    shp_paths = sorted(glob.glob(os.path.join(start_point_dir, '*.shp')))
    if not shp_paths:
        raise FileNotFoundError(f'no point .shp found in the candidate-site directory: {start_point_dir}')
    administrative_geometry = load_polygonal_area_geometry(administrative_boundary_shp)
    base_capacity = float(CLRP_DEPOT_CAPACITY)
    base_build_cost = float(CLRP_DEPOT_BUILD_COST)
    boundary_prj = os.path.splitext(administrative_boundary_shp)[0] + '.prj'
    if not os.path.exists(boundary_prj):
        raise FileNotFoundError(f'study-area shapefile is missing a CRS file: {boundary_prj}')
    target_crs = CRS.from_wkt(open(boundary_prj, encoding='utf-8-sig').read())
    no_fly = [poly for poly in prohibited_polygons or [] if not poly.is_empty]
    candidate_xy, seen, dropped = ([], set(), 0)
    for shp_path in shp_paths:
        prj_path = os.path.splitext(shp_path)[0] + '.prj'
        if not os.path.exists(prj_path):
            raise FileNotFoundError(f'candidate-site shapefile is missing a CRS file: {prj_path}')
        source_crs = CRS.from_wkt(open(prj_path, encoding='utf-8-sig').read())
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        for shape_record in shapefile.Reader(shp_path).shapes():
            for x, y in shape_record.points:
                x, y = transformer.transform(float(x), float(y))
                key = (round(x, 3), round(y, 3))
                point = Point(x, y)
                if key in seen or not administrative_geometry.covers(point) or any((poly.covers(point) for poly in no_fly)):
                    dropped += 1
                    continue
                seen.add(key)
                candidate_xy.append((x, y))
    if not candidate_xy:
        raise ValueError('no candidate sites remain after study-area / no-fly filtering')
    print(f'candidate-site files: {len(shp_paths)}; usable points: {len(candidate_xy)}; dropped duplicates/invalid: {dropped}')
    generated = []
    for idx, (x, y) in enumerate(candidate_xy):
        generated.append([float(idx), float(x), float(y), base_capacity, base_build_cost, float(CLRP_DEPOT_EXTRA_VALUE)])
    return generated

def generate_hexagonal_customer_data(administrative_boundary_shp, prohibited_polygons=None, hex_side_length=CLRP_WAYPOINT_CIRCLE_RADIUS, customer_demand=CLRP_CUSTOMER_DEMAND, farmland_shp=_FARMLAND_SHP):
    radius = float(hex_side_length)
    if radius <= 0:
        raise ValueError('CLRP_WAYPOINT_CIRCLE_RADIUS must be greater than 0')
    administrative_geometry = load_polygonal_area_geometry(administrative_boundary_shp)
    feasible = build_candidate_feasible_area(administrative_geometry, prohibited_polygons)
    target_crs = _require_metric_projected_crs(_read_prj_crs(administrative_boundary_shp), 'study-area CRS')
    farmland = load_polygonal_geometry_reprojected(farmland_shp, target_crs)
    farmland_parts = [part for part in _iter_polygonal_parts(farmland) if part is not None and (not part.is_empty)]
    farmland_tree = STRtree(farmland_parts)
    prepared_feasible = prep(feasible)
    minx, miny, maxx, maxy = feasible.bounds
    # Hexagonal layout: adjacent centre spacing = R * √3 (row step x_step, odd rows offset by half a step).
    x_step = radius * math.sqrt(3.0)
    y_step = 1.5 * radius
    customers = []
    generated_count = 0
    row = 0
    y = miny
    while y <= maxy + 1e-09:
        x = minx + ((0.5 * x_step) if row % 2 else 0.0)
        while x <= maxx + 1e-09:
            center = Point(x, y)
            if prepared_feasible.covers(center):
                generated_count += 1
                coverage_circle = center.buffer(radius)
                intersects_farmland = False
                for farmland_idx in farmland_tree.query(coverage_circle, predicate='intersects'):
                    if coverage_circle.intersects(farmland_parts[int(farmland_idx)]):
                        intersects_farmland = True
                        break
                if intersects_farmland:
                    customers.append([float(len(customers)), float(x), float(y), float(customer_demand)])
            x += x_step
        y += y_step
        row += 1
    if not customers:
        raise ValueError('no waypoints intersect farmland polygons; check CRS, polygon file, or circle radius')
    print(f'coverage radius: {radius:.1f} m; adjacent centre spacing: {x_step:.1f} m (R√3); row step: {x_step:.1f} m; row spacing: {y_step:.1f} m')
    print(f'free-flight candidate centres: {generated_count}; retained after farmland intersection: {len(customers)}; dropped: {generated_count - len(customers)}')
    return customers

def _build_clrp_arrays(customer_data, depot_data, n_selected_depots, max_vehicles_per_depot, vehicle_cap, penalize_long_edges=False):
    n_depot_candidates = len(depot_data)
    n_customers = len(customer_data)
    depot_candidates = list(range(n_depot_candidates))
    customers = list(range(n_depot_candidates, n_depot_candidates + n_customers))
    all_nodes = depot_candidates + customers
    coords = np.zeros((len(all_nodes), 2))
    depot_build_cost = {}
    depot_capacities = {}
    for idx, depot in enumerate(depot_data):
        node = depot_candidates[idx]
        coords[node] = [depot[1], depot[2]]
        depot_capacities[node] = depot[3]
        depot_build_cost[node] = depot[4]
    demands = np.zeros(len(all_nodes))
    for idx, customer in enumerate(customer_data):
        node = customers[idx]
        coords[node] = [customer[1], customer[2]]
        demands[node] = customer[3]
    if vehicle_cap is None:
        vehicle_cap = int(np.mean(list(depot_capacities.values())))
    else:
        vehicle_cap = int(vehicle_cap)
    total_demand = float(sum(demands))
    # With no open-depot count cap, only check the theoretical capacity of all candidate depots together.
    max_total_cap = n_depot_candidates * max_vehicles_per_depot * vehicle_cap
    if total_demand > max_total_cap:
        raise ValueError(f'total demand {total_demand:.0f} exceeds the theoretical capacity {max_total_cap:.0f} of all candidate depots; add candidates, raise routes-per-depot, or raise route capacity')
    dist_matrix = np.zeros((len(all_nodes), len(all_nodes)))
    for i in all_nodes:
        for j in all_nodes:
            if i == j:
                continue
            distance = max(np.linalg.norm(coords[i] - coords[j]), 0.1)
            if penalize_long_edges and distance > 10000.0:
                distance = 10000.0 + (distance - 10000.0) * 100.0
            dist_matrix[i][j] = distance
    return (n_depot_candidates, n_selected_depots, n_customers, max_vehicles_per_depot, vehicle_cap, depot_candidates, customers, all_nodes, coords, depot_build_cost, demands, dist_matrix)

def load_zunhua_UAV_data_for_CLRP(administrative_boundary_shp=_ADMIN_BOUNDARY_SHP, prohibited_polygons=None, route_prohibited_polygons=None, n_selected_depots=CLRP_N_SELECTED_DEPOTS, max_vehicles_per_depot=50, vehicle_cap=100, max_route_length=CLRP_MAX_ROUTE_LENGTH):
    print('=== Loading Zunhua UAV dataset (DEM + line-of-sight) ===')
    print(f'administrative feasible-area file (working projection): {administrative_boundary_shp}')
    print(f'original WGS84 study-area boundary: {_ADMIN_BOUNDARY_SHP_WGS84}')
    print(f'farmland filter shapefile: {_FARMLAND_SHP}')
    print(f'DEM file: {_DEM_TIF}')
    customer_data = generate_hexagonal_customer_data(administrative_boundary_shp, prohibited_polygons, hex_side_length=CLRP_WAYPOINT_CIRCLE_RADIUS, farmland_shp=_FARMLAND_SHP)
    generated_depot_data = load_candidate_depots_from_starting_points(administrative_boundary_shp=administrative_boundary_shp, prohibited_polygons=prohibited_polygons)
    route_polygons = prohibited_polygons if route_prohibited_polygons is None else route_prohibited_polygons
    target_crs = _require_metric_projected_crs(_read_prj_crs(administrative_boundary_shp), 'study-area CRS')
    dem_model = DEMTerrainModel(_DEM_TIF, target_crs)
    depot_xy_all = np.asarray([[row[1], row[2]] for row in generated_depot_data], dtype=float)
    depot_ground_all = dem_model.sample_many(depot_xy_all)
    valid_depot_mask = np.isfinite(depot_ground_all)
    dropped_depots = int((~valid_depot_mask).sum())
    if dropped_depots:
        print(f'[invalid depot DEM] dropped {dropped_depots} depots (outside DEM or NODATA).')
    generated_depot_data = [row for row, keep in zip(generated_depot_data, valid_depot_mask) if keep]
    depot_ground = depot_ground_all[valid_depot_mask]
    if not generated_depot_data:
        raise ValueError('no candidate depot has a valid DEM elevation; cannot build the instance.')
    generated_depot_data = [[float(i), *row[1:]] for i, row in enumerate(generated_depot_data)]
    depot_xy = np.asarray([[row[1], row[2]] for row in generated_depot_data], dtype=float)
    depot_height = depot_ground + CLRP_DEPOT_HEIGHT_AGL
    customer_xy_all = np.asarray([[row[1], row[2]] for row in customer_data], dtype=float)
    customer_ground_all = dem_model.sample_many(customer_xy_all)
    customer_height_all = customer_ground_all + CLRP_WAYPOINT_HEIGHT_AGL
    blocks = _free_flight_blocks_geometry(administrative_boundary_shp, route_polygons)
    depot_block = _point_block_ids(depot_xy, blocks)
    customer_block = _point_block_ids(customer_xy_all, blocks)
    depots_by_block = {}
    for depot_idx, block_id in enumerate(depot_block):
        if block_id >= 0:
            depots_by_block.setdefault(int(block_id), []).append(depot_idx)
    visibility_key = _matrix_checkpoint_key('visibility_AB', arrays=(depot_xy, customer_xy_all, depot_height, customer_height_all, depot_block, customer_block), config={'max_visibility_distance': float(CLRP_MAX_VISIBILITY_DISTANCE), 'profile_step': float(CLRP_DEM_PROFILE_STEP), 'intersection_tolerance': float(CLRP_TERRAIN_INTERSECTION_TOLERANCE), 'depot_agl': float(CLRP_DEPOT_HEIGHT_AGL), 'waypoint_agl': float(CLRP_WAYPOINT_HEIGHT_AGL)}, source_paths=(administrative_boundary_shp, _FARMLAND_SHP, _DEM_TIF, _WAYPOINT_EXCLUSION_SHP))
    cached_visibility = _load_matrix_checkpoint('visibility_AB', visibility_key, expected_shapes={'visibility_all': (len(generated_depot_data), len(customer_data))})
    if cached_visibility is not None:
        cached_arrays, cached_metadata = cached_visibility
        visibility_all = cached_arrays['visibility_all'].astype(bool)
        keep_customer_indices = cached_arrays['keep_customer_indices'].astype(int).tolist()
        details = list(cached_metadata.get('details', []))
        reason_counter = dict(cached_metadata.get('reason_counter', {}))
    else:
        visibility_all = np.zeros((len(generated_depot_data), len(customer_data)), dtype=bool)
        details = []
        keep_customer_indices = []
        reason_counter = {}
        for cust_idx, row in enumerate(customer_data):
            x, y = (float(row[1]), float(row[2]))
            reason = None
            block_id = int(customer_block[cust_idx])
            if not np.isfinite(customer_ground_all[cust_idx]):
                reason = 'WAYPOINT_DEM_NODATA'
            elif block_id < 0 or not depots_by_block.get(block_id):
                reason = 'NO_DEPOT_IN_BLOCK'
            else:
                within_range = 0
                terrain_blocked = 0
                for depot_idx in depots_by_block[block_id]:
                    visible, pair_reason, _ = dem_model.line_of_sight(depot_xy[depot_idx], customer_xy_all[cust_idx], depot_height[depot_idx], customer_height_all[cust_idx], max_distance=CLRP_MAX_VISIBILITY_DISTANCE)
                    if pair_reason != 'OUT_OF_RANGE':
                        within_range += 1
                    if pair_reason == 'TERRAIN_BLOCKED':
                        terrain_blocked += 1
                    if visible:
                        visibility_all[depot_idx, cust_idx] = True
                if np.any(visibility_all[:, cust_idx]):
                    keep_customer_indices.append(cust_idx)
                elif within_range == 0:
                    reason = 'OUT_OF_RANGE'
                elif terrain_blocked > 0:
                    reason = 'TERRAIN_BLOCKED'
                else:
                    reason = 'NO_VISIBLE_DEPOT'
            if reason is not None:
                reason_counter[reason] = reason_counter.get(reason, 0) + 1
                details.append({'original_waypoint_id': int(row[0]), 'x': x, 'y': y, 'reason': reason, 'block_id': block_id})
        _save_matrix_checkpoint('visibility_AB', visibility_key, arrays={'visibility_all': visibility_all.astype(np.uint8), 'keep_customer_indices': np.asarray(keep_customer_indices, dtype=np.int64)}, metadata={'details': details, 'reason_counter': reason_counter, 'shape': [len(generated_depot_data), len(customer_data)], 'description': 'A×B depot–raw-waypoint visibility matrix and reachable waypoint indices'})
    customer_data = [customer_data[i] for i in keep_customer_indices]
    customer_data = [[float(i), *row[1:]] for i, row in enumerate(customer_data)]
    customer_ground = customer_ground_all[keep_customer_indices]
    visibility = visibility_all[:, keep_customer_indices]
    if not customer_data:
        raise ValueError('all waypoints are unreachable under DEM or line-of-sight constraints; cannot build the instance.')
    global _LAST_UNPLANNABLE_WAYPOINTS, _LAST_PARTITION_UNPLANNABLE_WAYPOINTS
    global _LAST_UNPLANNABLE_DETAILS, _LAST_NODE_GROUND_ELEVATIONS
    global _LAST_NODE_FLIGHT_HEIGHTS, _LAST_DEPOT_WAYPOINT_VISIBILITY
    _LAST_UNPLANNABLE_DETAILS = details
    _LAST_UNPLANNABLE_WAYPOINTS = [(item['x'], item['y']) for item in details if item['reason'] != 'NO_DEPOT_IN_BLOCK']
    _LAST_PARTITION_UNPLANNABLE_WAYPOINTS = [(item['x'], item['y']) for item in details if item['reason'] == 'NO_DEPOT_IN_BLOCK']
    if reason_counter:
        reason_text = '; '.join((f'{key}={value}' for key, value in sorted(reason_counter.items())))
        print(f'[unreachable waypoints] dropped {len(details)}: {reason_text}')
        for item in details[:20]:
            print(f"  waypoint {item['original_waypoint_id']} ({item['x']:.2f}, {item['y']:.2f}): {item['reason']}")
        if len(details) > 20:
            print(f'  ... remaining {len(details) - 20} listed in the unreachable-waypoint CSV.')
    result = _build_clrp_arrays(customer_data, generated_depot_data, n_selected_depots, max_vehicles_per_depot, vehicle_cap, penalize_long_edges=False)
    n_depots, _, n_customers = result[:3]
    _LAST_NODE_GROUND_ELEVATIONS = np.r_[depot_ground, customer_ground]
    _LAST_NODE_FLIGHT_HEIGHTS = np.r_[depot_ground + CLRP_DEPOT_HEIGHT_AGL, customer_ground + CLRP_WAYPOINT_HEIGHT_AGL]
    full_visibility = np.zeros((n_depots + n_customers, n_depots + n_customers), dtype=bool)
    full_visibility[:n_depots, n_depots:] = visibility
    full_visibility[n_depots:, :n_depots] = visibility.T
    _LAST_DEPOT_WAYPOINT_VISIBILITY = full_visibility
    print('=== Zunhua UAV dataset loaded ===')
    print(f'candidate take-off/landing sites: {result[0]}')
    print(f'waypoints remaining after DEM and line-of-sight filtering: {result[2]}')
    print(f'depot height: DEM+{CLRP_DEPOT_HEIGHT_AGL:g} m; waypoint height: DEM+{CLRP_WAYPOINT_HEIGHT_AGL:g} m; LOS range: {CLRP_MAX_VISIBILITY_DISTANCE / 1000:g} km')
    print(f'max vehicles per depot: {result[3]}')
    print(f'vehicle capacity: {result[4]}')
    print(f'total customer demand: {sum(result[10]):.0f}')
    return result

def load_prohibited_flight_areas(shp_path):
    print('=== Loading no-fly-zone data ===')
    if shp_path is None or not str(shp_path).strip():
        print('no no-fly shapefile provided: treat the whole area as flyable.\n')
        return []
    shp_path = os.fspath(shp_path)
    print(f'no-fly shapefile: {shp_path}')
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f'no-fly shapefile does not exist: {shp_path}. Pass None or an empty string to disable no-fly zones.')
    sf = shapefile.Reader(shp_path)
    source_prj = os.path.splitext(shp_path)[0] + '.prj'
    target_prj = os.path.splitext(_ADMIN_BOUNDARY_SHP)[0] + '.prj'
    if not os.path.exists(source_prj) or not os.path.exists(target_prj):
        raise FileNotFoundError('no-fly or administrative boundary is missing a .prj CRS file')
    source_crs = CRS.from_wkt(open(source_prj, encoding='utf-8-sig').read())
    target_crs = CRS.from_wkt(open(target_prj, encoding='utf-8-sig').read())
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    polygons = []
    is_railway = os.path.normcase(os.path.abspath(shp_path)) == os.path.normcase(os.path.abspath(_RAILWAY_SHP))
    study_area = load_polygonal_area_geometry(_ADMIN_BOUNDARY_SHP) if is_railway else None
    for i, shp_obj in enumerate(sf.shapes()):
        geometry = shapely_shape(shp_obj.__geo_interface__)
        geometry = shapely_transform(transformer.transform, geometry)
        if is_railway:
            geometry = geometry.intersection(study_area)
            if geometry.is_empty:
                continue
            geometry = geometry.buffer(CLRP_RAILWAY_BUFFER_DISTANCE).intersection(study_area)
        for poly in _iter_polygonal_parts(geometry):
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty:
                polygons.append(poly)
        if i % 1000 == 0 and i > 0:
            print(f'  processed {i} railway features...')
    if is_railway and polygons:
        merged_railway_buffer = unary_union(polygons)
        if not merged_railway_buffer.is_valid:
            merged_railway_buffer = merged_railway_buffer.buffer(0)
        polygons = list(_iter_polygonal_parts(merged_railway_buffer))
    label = f'railway buffer no-fly zone ({CLRP_RAILWAY_BUFFER_DISTANCE:g} m on each side)' if is_railway else 'no-fly zone'
    print(f'loaded {len(polygons)} {label} polygon(s)\n')
    return polygons

def load_administrative_boundaries(shp_path):
    if not shp_path or not os.path.exists(shp_path):
        return []
    sf = shapefile.Reader(shp_path)
    boundary_parts = []
    for shp in sf.shapes():
        points = np.asarray(shp.points, dtype=float)
        if len(points) < 2:
            continue
        part_starts = list(shp.parts) + [len(points)]
        for start, end in zip(part_starts[:-1], part_starts[1:]):
            if end - start >= 2:
                boundary_parts.append(points[start:end])
    print(f'administrative boundary: {os.path.basename(shp_path)}, {len(boundary_parts)} part(s)')
    return boundary_parts

def build_no_fly_crossing_matrix(all_nodes, coords, prohibited_polygons):
    n = len(all_nodes)
    crossing = np.zeros((n, n), dtype=bool)
    if not prohibited_polygons:
        print(f'=== no no-fly zone: skip {n}×{n} crossing matrix, whole area flyable ===\n')
        return crossing
    n_prohibited = 0
    n_total = 0
    print(f'=== Precomputing no-fly crossing matrix ({n}×{n} node pairs) ===')
    no_fly_tree = STRtree(prohibited_polygons)
    for i in range(n):
        for j in range(i + 1, n):
            n_total += 1
            line = LineString([(coords[i][0], coords[i][1]), (coords[j][0], coords[j][1])])
            for poly_idx in no_fly_tree.query(line, predicate='intersects'):
                poly = prohibited_polygons[int(poly_idx)]
                if line.intersects(poly) and (not line.touches(poly)):
                    crossing[i][j] = True
                    crossing[j][i] = True
                    n_prohibited += 1
                    break
        if i % 100 == 0 and i > 0:
            print(f'  processed {i}/{n} nodes...')
    pct = n_prohibited / max(n_total, 1) * 100
    print(f'edges crossing a no-fly zone: {n_prohibited}/{n_total} ({pct:.1f}%)\n')
    return crossing

class ALNS_CLRP:

    def __init__(self, depot_candidates, customers, all_nodes, coords, depot_build_cost, demands, vehicle_cap, dist_matrix, n_selected_depots, max_vehicles_per_depot, max_route_length=CLRP_MAX_ROUTE_LENGTH, no_fly_crossing=None, navigation_distances=None, visibility_matrix=None, ground_elevations=None, flight_heights=None, ascent_distances=None, horizontal_distances=None, descent_distances=None, vertical_distances=None, cruise_heights=None):
        self.depot_candidates = depot_candidates
        self.customers = customers
        self.all_nodes = all_nodes
        self.coords = coords
        self.depot_build_cost = depot_build_cost
        self.demands = demands
        self.vehicle_cap = vehicle_cap
        self.dist_matrix = dist_matrix
        # Keep the existing call interface; open-depot count is no longer bounded by min/max parameters.
        self.n_selected_depots = n_selected_depots
        self.max_veh_per_depot = max_vehicles_per_depot
        self.max_route_length = max_route_length
        self.no_fly_crossing = no_fly_crossing if no_fly_crossing is not None else np.zeros((len(all_nodes), len(all_nodes)), dtype=bool)
        n_nodes = len(all_nodes)
        self.visibility_matrix = np.asarray(visibility_matrix, dtype=bool) if visibility_matrix is not None else np.ones((n_nodes, n_nodes), dtype=bool)
        self.ground_elevations = None if ground_elevations is None else np.asarray(ground_elevations, dtype=float)
        self.flight_heights = None if flight_heights is None else np.asarray(flight_heights, dtype=float)
        self.ascent_distances = None if ascent_distances is None else np.asarray(ascent_distances, dtype=np.float32)
        self.horizontal_distances = None if horizontal_distances is None else np.asarray(horizontal_distances, dtype=np.float32)
        self.descent_distances = None if descent_distances is None else np.asarray(descent_distances, dtype=np.float32)
        self.vertical_distances = None if vertical_distances is None else np.asarray(vertical_distances, dtype=np.float32)
        self.cruise_heights = None if cruise_heights is None else np.asarray(cruise_heights, dtype=np.float32)
        n_all = len(all_nodes)
        self.actual_dist = np.zeros((n_all, n_all), dtype=np.float32)
        for i in range(n_all):
            for j in range(n_all):
                if i != j:
                    self.actual_dist[i][j] = max(np.linalg.norm(coords[i] - coords[j]), 0.1)
        has_directional_matrices = all((matrix is not None for matrix in (self.ascent_distances, self.horizontal_distances, self.descent_distances)))
        if has_directional_matrices:
            # Total distance is always rebuilt from the three directed components in one float32 order.
            self.actual_dist = _compose_total_flight_distance(
                self.ascent_distances,
                self.horizontal_distances,
                self.descent_distances,
            )
            self.vertical_distances = np.add(
                self.ascent_distances,
                self.descent_distances,
                dtype=np.float32,
            )
        elif navigation_distances is not None:
            self.actual_dist = np.asarray(navigation_distances, dtype=np.float32)
        if navigation_distances is not None:
            supplied_total = np.asarray(navigation_distances, dtype=np.float32)
            report = _distance_matrix_consistency_report(supplied_total, self.actual_dist)
            if report.get('shape_mismatch'):
                raise ValueError(
                    'navigation_distances shape does not match the node matrix: '
                    f"supplied={report['supplied_shape']}, rebuilt={report['rebuilt_shape']}"
                )
            if has_directional_matrices and not report['consistent']:
                raise ValueError(
                    'total distance matrix is materially inconsistent with ascent+horizontal+descent: '
                    f"finite-pattern mismatch={report['finite_pattern_mismatch']}, "
                    f"out-of-tolerance entries={report['mismatch_count']}, "
                    f"max error={report['max_difference']:.9f} m, "
                    f"mean error={report['mean_difference']:.9f} m, "
                    f"rtol={CLRP_DISTANCE_MATRIX_CHECK_RTOL:g}, "
                    f"atol={CLRP_DISTANCE_MATRIX_CHECK_ATOL:g} m"
                )
            self.no_fly_crossing = np.zeros_like(self.no_fly_crossing, dtype=bool)
        self.dist_matrix = np.asarray(self.actual_dist, dtype=np.float32).copy()
        self.customer_feasible_depots = {}
        self.depot_cover_sets = {d: set() for d in self.depot_candidates}
        self._depot_position = {d: idx for idx, d in enumerate(self.depot_candidates)}
        self._customer_position = {c: idx for idx, c in enumerate(self.customers)}
        self._initial_service_cost = np.full((len(self.depot_candidates), len(self.customers)), np.inf, dtype=np.float32)
        # Each solver receives one free-flight block, so nodes do not need another O(n²) connectivity scan.
        self.node_component = {node: 0 for node in self.all_nodes}

        # Depot–waypoint service feasibility uses matrix operations instead of millions of Python nested loops.
        depot_index = np.asarray(self.depot_candidates, dtype=int)
        customer_index = np.asarray(self.customers, dtype=int)
        service_mask = self.visibility_matrix[np.ix_(depot_index, customer_index)].copy()
        service_mask &= np.isfinite(self.actual_dist[np.ix_(depot_index, customer_index)])
        service_mask &= (2.0 * self.actual_dist[np.ix_(depot_index, customer_index)]
                         <= self.max_route_length + 1e-10)
        self.direct_gateway_customers = {
            int(depot): {int(self.customers[pos]) for pos in np.flatnonzero(service_mask[row])}
            for row, depot in enumerate(self.depot_candidates)
        }
        self.component_gateway_depots = {}
        for depot, gateway_customers in self.direct_gateway_customers.items():
            if gateway_customers:
                component = self.node_component[depot]
                self.component_gateway_depots.setdefault(component, []).append(depot)
        for cust in self.customers:
            feasible = [d for d in self.depot_candidates if cust in self.direct_gateway_customers[d]]
            if not feasible:
                raise ValueError(f'customer {cust} cannot be served round-trip within {self.max_route_length / 1000:g} km from any candidate depot under the current detour distances; the instance is infeasible.')
            self.customer_feasible_depots[cust] = tuple(feasible)
            cust_pos = self._customer_position[cust]
            for depot in feasible:
                self.depot_cover_sets[depot].add(cust)
                self._initial_service_cost[self._depot_position[depot], cust_pos] = 2.0 * self.dist_matrix[depot][cust]
        self.selected_depots = None
        self.best_routes = {}
        self.best_build_cost = 0
        self.best_transport_cost = 0
        self.best_total_cost = float('inf')

    def can_depot_serve_customer(self, depot, customer):
        if depot not in self.depot_candidates or customer not in self.customers:
            return False
        return bool(self.visibility_matrix[int(depot), int(customer)])

    def check_route_visibility(self, route, depot=None):
        if depot is None:
            depot = route[0]
        return all((self.can_depot_serve_customer(depot, node) for node in route[1:-1] if node in self.customers))

    def edge_flight_distance(self, a, b):
        if self.ascent_distances is not None and self.horizontal_distances is not None and (self.descent_distances is not None):
            if _ablation_planar_distance():
                return float(self.horizontal_distances[a][b])
            return float(self.ascent_distances[a][b] + self.horizontal_distances[a][b] + self.descent_distances[a][b])
        return float(self.actual_dist[a][b])

    def calculate_route_cost(self, route):
        return sum((self.edge_flight_distance(a, b) for a, b in zip(route[:-1], route[1:])))

    def calculate_total_transport_cost(self, routes_dict):
        total = 0
        for depot in routes_dict:
            for route in routes_dict[depot]:
                total += self.calculate_route_cost(route) + CLRP_VEHICLE_FIXED_COST
        return total

    def calculate_build_cost(self, selected_depots):
        return sum((self.depot_build_cost[j] for j in selected_depots))

    def check_route_capacity(self, route):
        total_demand = sum((self.demands[node] for node in route if node in self.customers))
        return total_demand <= self.vehicle_cap

    def calculate_route_length_actual(self, route):
        return sum((self.edge_flight_distance(a, b) for a, b in zip(route[:-1], route[1:])))

    def check_route_length(self, route):
        return self.calculate_route_length_actual(route) <= self.max_route_length

    def edge_crosses_no_fly(self, a, b):
        return self.no_fly_crossing[a][b]

    def validate_solution(self, routes_dict, require_all_customers=True):
        violations = []
        visits = {cust: 0 for cust in self.customers}
        for depot, routes in routes_dict.items():
            if depot not in self.depot_candidates:
                violations.append(f'node {depot} is not a candidate depot')
            if len(routes) > self.max_veh_per_depot:
                violations.append(f'depot {depot} uses {len(routes)} vehicles, limit={self.max_veh_per_depot}')
            for route_idx, route in enumerate(routes):
                route_name = f'depot {depot} route {route_idx + 1}'
                if len(route) < 3 or route[0] != depot or route[-1] != depot:
                    violations.append(f'{route_name}does not start and end at its assigned depot')
                    continue
                if not self.check_route_capacity(route):
                    violations.append(f'{route_name}exceeds vehicle capacity')
                if not self.check_route_length(route):
                    violations.append(f'{route_name}exceeds maximum route length')
                if not self.check_route_visibility(route, depot):
                    invisible = [node for node in route[1:-1] if node in self.customers and (not self.can_depot_serve_customer(depot, node))]
                    violations.append(f'{route_name}contains {len(invisible)} waypoint(s) without line of sight to the depot: {invisible[:10]}')
                for a, b in zip(route[:-1], route[1:]):
                    if self.edge_crosses_no_fly(a, b):
                        violations.append(f'{route_name} leg ({a},{b}) crosses a no-fly zone')
                for node in route[1:-1]:
                    if node in visits:
                        visits[node] += 1
                    else:
                        violations.append(f'{route_name}contains illegal customer node {node}')
        if require_all_customers:
            missing = [cust for cust, count in visits.items() if count == 0]
            duplicate = [cust for cust, count in visits.items() if count > 1]
            if missing:
                violations.append(f'{len(missing)} customer(s) are unserved')
            if duplicate:
                violations.append(f'{len(duplicate)} customer(s) are served more than once')
        return (not violations, violations)

    def _selected_depots_cover_all(self, selected_depots):
        selected = set(selected_depots)
        if not selected:
            return False
        rows = [self._depot_position[d] for d in selected]
        nearest = np.min(self._initial_service_cost[rows, :], axis=0)
        return bool(np.all(np.isfinite(nearest)))

    def _initial_depot_proxy_cost(self, selected_depots):
        selected = set(selected_depots)
        if not selected:
            return float('inf')
        rows = [self._depot_position[d] for d in selected]
        nearest = np.min(self._initial_service_cost[rows, :], axis=0)
        if not np.all(np.isfinite(nearest)):
            return float('inf')
        return float(self.calculate_build_cost(selected) + np.sum(nearest))

    def _improve_initial_depot_selection(self, selected_depots):
        selected = set(selected_depots)
        current_cost = self._initial_depot_proxy_cost(selected)
        for _ in range(CLRP_INITIAL_SWAP_PASSES):
            improved = False
            selected_order = list(selected)
            unopened_order = [d for d in self.depot_candidates if d not in selected]
            random.shuffle(selected_order)
            random.shuffle(unopened_order)
            for depot_out in selected_order:
                for depot_in in unopened_order:
                    candidate = selected - {depot_out} | {depot_in}
                    if not self._selected_depots_cover_all(candidate):
                        continue
                    candidate_cost = self._initial_depot_proxy_cost(candidate)
                    if candidate_cost < current_cost - 1e-09:
                        selected = candidate
                        current_cost = candidate_cost
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return (list(selected), current_cost)

    def _construct_visibility_greedy_depots(self, target_p=None):
        all_customers = set(self.customers)
        mandatory = {feasible[0] for feasible in self.customer_feasible_depots.values() if len(feasible) == 1}
        selected = set(mandatory)
        covered = set().union(*(self.depot_cover_sets[d] for d in selected)) if selected else set()
        uncovered = all_customers - covered
        while uncovered:
            remaining = [d for d in self.depot_candidates if d not in selected]
            scored = []
            for depot in remaining:
                newly = self.depot_cover_sets[depot] & uncovered
                if not newly:
                    continue
                mean_cost = float(np.mean([self.dist_matrix[depot][c] for c in newly]))
                scored.append((-len(newly), mean_cost, float(self.depot_build_cost[depot]), depot))
            if not scored:
                raise ValueError(f'visibility-greedy siting still leaves {len(uncovered)} waypoint(s) uncovered')
            _, _, _, chosen = min(scored)
            selected.add(chosen)
            covered.update(self.depot_cover_sets[chosen])
            uncovered = all_customers - covered
        if target_p is not None:
            target_p = int(target_p)
            if len(selected) > target_p:
                return None
            while len(selected) < target_p:
                remaining = [d for d in self.depot_candidates if d not in selected]
                if not remaining:
                    return None
                chosen = min(remaining, key=lambda d: (self._initial_depot_proxy_cost(selected | {d}), self.depot_build_cost[d], d))
                selected.add(chosen)
        improved, _ = self._improve_initial_depot_selection(selected)
        return list(improved)

    def _construct_visibility_greedy_routes(self, selected_depots):
        selected = sorted(set(selected_depots))
        routes_dict = {depot: [] for depot in selected}
        customer_order = sorted(self.customers, key=lambda customer: (sum((self.can_depot_serve_customer(depot, customer) for depot in selected)), -float(self.demands[customer]), -min((2.0 * self.edge_flight_distance(depot, customer) for depot in selected if self.can_depot_serve_customer(depot, customer)), default=float('inf')), customer))
        for customer in customer_order:
            candidates = []
            for depot in selected:
                if not self.can_depot_serve_customer(depot, customer):
                    continue
                for route_idx, route in enumerate(routes_dict[depot]):
                    if sum((self.demands[node] for node in route[1:-1])) + self.demands[customer] > self.vehicle_cap + 1e-10:
                        continue
                    for position in range(1, len(route)):
                        prev_node, next_node = (route[position - 1], route[position])
                        if self.edge_crosses_no_fly(prev_node, customer) or self.edge_crosses_no_fly(customer, next_node):
                            continue
                        candidate_route = route[:position] + [customer] + route[position:]
                        if not self.check_route_length(candidate_route):
                            continue
                        delta = self.edge_flight_distance(prev_node, customer) + self.edge_flight_distance(customer, next_node) - self.edge_flight_distance(prev_node, next_node)
                        candidates.append((delta, 0, depot, route_idx, position, candidate_route))
                if len(routes_dict[depot]) < self.max_veh_per_depot:
                    candidate_route = [depot, customer, depot]
                    if not self.edge_crosses_no_fly(depot, customer) and self.check_route_length(candidate_route) and self.check_route_capacity(candidate_route):
                        delta = self.calculate_route_cost(candidate_route) + CLRP_VEHICLE_FIXED_COST
                        candidates.append((delta, 1, depot, len(routes_dict[depot]), 1, candidate_route))
            if not candidates:
                visible_depots = [depot for depot in selected if self.can_depot_serve_customer(depot, customer)]
                reason = 'no selected depot has line of sight' if not visible_depots else 'visible depots are all blocked by range, capacity, vehicle-count, or no-fly edges'
                return (None, f'waypoint {customer} cannot be greedily inserted: {reason}')
            _, is_new_route, depot, route_idx, _, candidate_route = min(candidates)
            if is_new_route:
                routes_dict[depot].append(candidate_route)
            else:
                routes_dict[depot][route_idx] = candidate_route
        feasible, violations = self.validate_solution(routes_dict)
        if not feasible:
            return (None, '; '.join(violations[:8]))
        return (routes_dict, None)

    def _construct_routes_for_customer_subset(self, selected_depots, customer_subset):
        """Build closed routes only for a given waypoint subset; other waypoints in the block need not be covered."""
        selected = sorted(set(selected_depots))
        customers = sorted(set(customer_subset))
        if not selected or not customers:
            return (None, 'local depot set or waypoint set is empty')
        routes_dict = {depot: [] for depot in selected}
        customer_order = sorted(
            customers,
            key=lambda customer: (
                sum(self.can_depot_serve_customer(depot, customer) for depot in selected),
                -float(self.demands[customer]),
                -min((2.0 * self.edge_flight_distance(depot, customer)
                      for depot in selected if self.can_depot_serve_customer(depot, customer)),
                     default=float('inf')),
                customer,
            ),
        )
        for customer in customer_order:
            candidates = []
            for depot in selected:
                if not self.can_depot_serve_customer(depot, customer):
                    continue
                for route_idx, route in enumerate(routes_dict[depot]):
                    route_load = sum(self.demands[node] for node in route[1:-1])
                    if route_load + self.demands[customer] > self.vehicle_cap + 1e-10:
                        continue
                    for position in range(1, len(route)):
                        prev_node, next_node = route[position - 1], route[position]
                        if self.edge_crosses_no_fly(prev_node, customer) or self.edge_crosses_no_fly(customer, next_node):
                            continue
                        candidate_route = route[:position] + [customer] + route[position:]
                        if not self.check_route_length(candidate_route):
                            continue
                        delta = (self.edge_flight_distance(prev_node, customer)
                                 + self.edge_flight_distance(customer, next_node)
                                 - self.edge_flight_distance(prev_node, next_node))
                        candidates.append((delta, 0, depot, route_idx, candidate_route))
                if len(routes_dict[depot]) < self.max_veh_per_depot:
                    candidate_route = [depot, customer, depot]
                    if (not self.edge_crosses_no_fly(depot, customer)
                            and self.check_route_length(candidate_route)
                            and self.check_route_capacity(candidate_route)):
                        delta = self.calculate_route_cost(candidate_route) + CLRP_VEHICLE_FIXED_COST
                        candidates.append((delta, 1, depot, len(routes_dict[depot]), candidate_route))
            if not candidates:
                return (None, f'local waypoint {customer} cannot be inserted into candidate depot set {selected}')
            _, is_new_route, depot, route_idx, candidate_route = min(candidates)
            if is_new_route:
                routes_dict[depot].append(candidate_route)
            else:
                routes_dict[depot][route_idx] = candidate_route

        # Run intra-route 2-opt/Or-opt only on newly built local routes; do not touch routes outside the region.
        for depot, depot_routes in routes_dict.items():
            optimized = []
            for route in depot_routes:
                candidate = self._two_opt_route(route)
                if hasattr(self, '_or_opt_route'):
                    candidate = self._or_opt_route(candidate)
                if not self._route_is_feasible(candidate, depot):
                    candidate = route
                optimized.append(candidate)
            routes_dict[depot] = optimized

        visits = {customer: 0 for customer in customers}
        for depot, depot_routes in routes_dict.items():
            if not depot_routes:
                return (None, f'candidate depot {depot} serves no waypoints; reject the empty-depot plan')
            if len(depot_routes) > self.max_veh_per_depot:
                return (None, f'candidate depot {depot} exceeds the route-count limit')
            for route in depot_routes:
                if not self._route_is_feasible(route, depot):
                    return (None, f'candidate depot {depot} has an infeasible local route')
                for customer in route[1:-1]:
                    if customer in visits:
                        visits[customer] += 1
        if any(count != 1 for count in visits.values()):
            return (None, 'local waypoints are not served exactly once')
        return (routes_dict, None)

    @staticmethod
    def _normalize_metric(values):
        if not values:
            return {}
        lo, hi = min(values.values()), max(values.values())
        if hi - lo <= 1e-12:
            return {key: 0.5 for key in values}
        return {key: (value - lo) / (hi - lo) for key, value in values.items()}

    def _assigned_customers_by_depot(self, routes_dict):
        return {
            depot: [customer for route in routes for customer in route[1:-1]]
            for depot, routes in routes_dict.items()
        }

    def _coverage_circle_union_area(self, customers, area_cache):
        key = tuple(sorted(set(customers)))
        if key in area_cache:
            return area_cache[key]
        if not key:
            area_cache[key] = 0.0
            return 0.0
        circles = [Point(float(self.coords[c][0]), float(self.coords[c][1])).buffer(CLRP_WAYPOINT_CIRCLE_RADIUS)
                   for c in key]
        area = float(unary_union(circles).area)
        area_cache[key] = area
        return area

    def _initial_facility_scores(self, routes_dict, area_cache):
        assigned = self._assigned_customers_by_depot(routes_dict)
        counts = {depot: float(len(customers)) for depot, customers in assigned.items()}
        areas = {depot: self._coverage_circle_union_area(customers, area_cache)
                 for depot, customers in assigned.items()}
        count_norm = self._normalize_metric(counts)
        area_norm = self._normalize_metric(areas)
        scores = {
            depot: (CLRP_INITIAL_FACILITY_COUNT_WEIGHT * count_norm[depot]
                    + CLRP_INITIAL_FACILITY_AREA_WEIGHT * area_norm[depot])
            for depot in assigned
        }
        return assigned, counts, areas, scores

    def _adjacent_open_depot_groups(self, routes_dict, group_size):
        depots = sorted(routes_dict)
        if len(depots) < group_size:
            return []
        k = min(CLRP_INITIAL_FACILITY_K_NEIGHBORS, len(depots) - 1)
        neighbor_map = {}
        for depot in depots:
            same_component = [other for other in depots if other != depot
                              and self.node_component.get(other) == self.node_component.get(depot)]
            same_component.sort(key=lambda other: (float(np.linalg.norm(self.coords[depot] - self.coords[other])), other))
            neighbor_map[depot] = same_component[:k]
        groups = set()
        if group_size == 1:
            component_counts = {}
            for depot in depots:
                component = self.node_component.get(depot)
                component_counts[component] = component_counts.get(component, 0) + 1
            for depot in depots:
                if component_counts.get(self.node_component.get(depot), 0) == 1:
                    groups.add((depot,))
        elif group_size == 2:
            for depot, neighbors in neighbor_map.items():
                for other in neighbors:
                    groups.add(tuple(sorted((depot, other))))
        elif group_size == 3:
            for depot, neighbors in neighbor_map.items():
                for pair in combinations(neighbors, 2):
                    groups.add(tuple(sorted((depot, *pair))))
        return sorted(groups)

    def _rank_initial_facility_groups(self, routes_dict, mode, area_cache):
        assigned, counts, areas, scores = self._initial_facility_scores(routes_dict, area_cache)
        if mode in ('relocate_1_to_1', 'expand_1_to_2'):
            group_size = 1
        elif mode == 'relocate_2_to_2':
            group_size = 2
        else:
            group_size = 3
        groups = self._adjacent_open_depot_groups(routes_dict, group_size)
        ranked = []
        for group in groups:
            local_customers = set().union(*(set(assigned[d]) for d in group))
            if not local_customers:
                continue
            compactness = max(float(np.linalg.norm(self.coords[a] - self.coords[b]))
                              for a, b in combinations(group, 2)) if len(group) > 1 else 0.0
            if mode in ('relocate_1_to_1', 'expand_1_to_2'):
                # When a partition has only one open depot, prefer partitions with larger coverage first.
                primary = -scores[group[0]]
            elif mode == 'relocate_2_to_2':
                primary = -abs(scores[group[0]] - scores[group[1]])
            elif mode == 'reduce_3_to_2':
                primary = sum(scores[d] for d in group) / len(group)
            else:
                primary = -sum(scores[d] for d in group) / len(group)
            # smaller primary is better; compact regions are a secondary tie-break.
            ranked.append((primary, compactness, group, local_customers,
                           sum(counts[d] for d in group), sum(areas[d] for d in group)))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return ranked

    def _local_replacement_candidate_pool(self, routes_dict, old_group, local_customers, allow_old):
        old_group = set(old_group)
        current_open = set(routes_dict)
        points = np.vstack([self.coords[list(old_group)], self.coords[list(local_customers)]])
        min_xy = points.min(axis=0) - CLRP_INITIAL_FACILITY_REGION_BUFFER
        max_xy = points.max(axis=0) + CLRP_INITIAL_FACILITY_REGION_BUFFER
        component_ids = {self.node_component.get(depot) for depot in old_group}
        candidates = []
        for depot in self.depot_candidates:
            if depot in current_open and depot not in old_group:
                continue
            if depot in old_group and not allow_old:
                continue
            if self.node_component.get(depot) not in component_ids:
                continue
            xy = self.coords[depot]
            if np.any(xy < min_xy) or np.any(xy > max_xy):
                continue
            covered = set(local_customers) & self.depot_cover_sets[depot]
            if not covered:
                continue
            mean_distance = float(np.mean([self.edge_flight_distance(depot, customer) for customer in covered]))
            candidates.append((-len(covered), mean_distance, float(self.depot_build_cost[depot]), depot))
        candidates.sort()
        pool = [item[-1] for item in candidates[:CLRP_INITIAL_FACILITY_CANDIDATE_POOL]]
        if allow_old:
            for depot in sorted(old_group):
                if depot not in pool:
                    pool.append(depot)
        return sorted(set(pool))

    def _replacement_combinations(self, routes_dict, old_group, local_customers, target_size, disjoint):
        old_group = set(old_group)
        pool = self._local_replacement_candidate_pool(
            routes_dict, old_group, local_customers, allow_old=not disjoint)
        if disjoint:
            pool = [depot for depot in pool if depot not in old_group]
        if len(pool) < target_size:
            return []
        total_demand = float(sum(self.demands[customer] for customer in local_customers))
        if total_demand > target_size * self.max_veh_per_depot * self.vehicle_cap + 1e-10:
            return []
        scored = []
        local_customers = set(local_customers)
        minimum_routes = math.ceil(total_demand / max(self.vehicle_cap, 1e-12))
        for combo in combinations(pool, target_size):
            combo_set = set(combo)
            if disjoint and combo_set & old_group:
                continue
            covered = set().union(*(self.depot_cover_sets[depot] for depot in combo_set))
            if not local_customers <= covered:
                continue
            nearest_sum = 0.0
            feasible = True
            for customer in local_customers:
                distances = [2.0 * self.edge_flight_distance(depot, customer)
                             for depot in combo_set if customer in self.depot_cover_sets[depot]]
                if not distances:
                    feasible = False
                    break
                nearest_sum += min(distances)
            if not feasible:
                continue
            proxy = (self.calculate_build_cost(combo_set) + nearest_sum
                     + minimum_routes * CLRP_VEHICLE_FIXED_COST)
            scored.append((proxy, tuple(sorted(combo))))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [combo for _, combo in scored[:CLRP_INITIAL_FACILITY_EXACT_TRIALS]]

    def _try_initial_facility_group(self, routes_dict, old_group, target_size, disjoint, min_relative_improvement=0.0):
        old_group = tuple(sorted(old_group))
        assigned = self._assigned_customers_by_depot(routes_dict)
        local_customers = set().union(*(set(assigned[depot]) for depot in old_group))
        if not local_customers:
            return None
        combinations_to_test = self._replacement_combinations(
            routes_dict, old_group, local_customers, target_size, disjoint)
        if not combinations_to_test:
            return None
        current_total = self.calculate_build_cost(routes_dict.keys()) + self.calculate_total_transport_cost(routes_dict)
        min_relative_improvement = max(0.0, float(min_relative_improvement))
        required_improvement = max(
            float(CLRP_INITIAL_FACILITY_MIN_IMPROVEMENT),
            current_total * min_relative_improvement)
        outside_routes = {depot: deepcopy(routes) for depot, routes in routes_dict.items() if depot not in old_group}
        best = None
        for replacement in combinations_to_test:
            local_routes, failure = self._construct_routes_for_customer_subset(replacement, local_customers)
            if local_routes is None:
                continue
            candidate = deepcopy(outside_routes)
            candidate.update(local_routes)
            feasible, _ = self.validate_solution(candidate)
            if not feasible:
                continue
            candidate_total = self.calculate_build_cost(candidate.keys()) + self.calculate_total_transport_cost(candidate)
            if candidate_total < current_total - required_improvement - 1e-10:
                if best is None or candidate_total < best[0] - 1e-10:
                    best = (candidate_total, candidate, replacement)
        return best

    def _try_initial_facility_operator(self, routes_dict, mode, rank_index, area_cache,
                                       min_relative_improvement=None, rank_limit=None):
        ranked_groups = self._rank_initial_facility_groups(routes_dict, mode, area_cache)
        if not ranked_groups:
            return None
        if rank_limit is None:
            rank_limit = CLRP_INITIAL_FACILITY_NO_IMPROVE_LIMIT
        rank_limit = min(len(ranked_groups), max(1, int(rank_limit)))
        selected_index = int(rank_index) % rank_limit
        _, _, group, _, point_count, area = ranked_groups[selected_index]
        if mode == 'relocate_1_to_1':
            target_size, disjoint = 1, True
        elif mode == 'expand_1_to_2':
            target_size, disjoint = 2, False
        elif mode == 'relocate_2_to_2':
            target_size, disjoint = 2, True
        elif mode == 'reduce_3_to_2':
            target_size, disjoint = 2, False
        elif mode == 'expand_3_to_4':
            target_size, disjoint = 4, False
        else:
            raise ValueError(f'unknown depot-structure operator: {mode}')
        if min_relative_improvement is None:
            min_relative_improvement = (
                CLRP_INITIAL_FACILITY_EXPAND_MIN_REL_IMPROVEMENT
                if mode in ('expand_1_to_2', 'expand_3_to_4') else 0.0)
        result = self._try_initial_facility_group(
            routes_dict, group, target_size, disjoint,
            min_relative_improvement=min_relative_improvement)
        if result is None:
            return None
        candidate_total, candidate_routes, replacement = result
        current_total = self.calculate_build_cost(routes_dict.keys()) + self.calculate_total_transport_cost(routes_dict)
        improvement_ratio = (current_total - candidate_total) / max(abs(current_total), 1e-12)
        return {
            'mode': mode,
            'old_group': tuple(group),
            'replacement': tuple(replacement),
            'candidate_total': float(candidate_total),
            'routes': candidate_routes,
            'point_count': int(point_count),
            'coverage_area': float(area),
            'improvement_ratio': float(improvement_ratio),
        }

    def _refine_initial_facility_structure(self, routes_dict):
        if (not CLRP_INITIAL_FACILITY_REFINEMENT) or _ablation_disable_facility_ops():
            return routes_dict
        print('\n=== Initial depot-structure refinement: 1→1 / 1→2 / 2→2 / 3→2 / 3→4 ===')
        current = deepcopy(routes_dict)
        current_cost = self.calculate_build_cost(current.keys()) + self.calculate_total_transport_cost(current)
        initial_cost = current_cost
        initial_depots = len(current)
        no_improve_passes = 0
        pass_index = 0
        accepted_moves = 0
        area_cache = {}
        modes = (
            'relocate_1_to_1',
            'relocate_2_to_2',
            'reduce_3_to_2',
            'expand_1_to_2',
            'expand_3_to_4',
        )
        mode_labels = {
            'relocate_1_to_1': 'single-depot relocate 1→1',
            'expand_1_to_2': 'single-depot expand 1→2',
            'relocate_2_to_2': 'relocate 2→2',
            'reduce_3_to_2': 'small-region merge 3→2',
            'expand_3_to_4': 'large-region expand 3→4',
        }
        self.initial_facility_refinement_history = []
        while (no_improve_passes < CLRP_INITIAL_FACILITY_NO_IMPROVE_LIMIT
               and pass_index < CLRP_INITIAL_FACILITY_MAX_PASSES):
            pass_index += 1
            improved_this_pass = False
            rank_index = no_improve_passes
            for mode in modes:
                result = self._try_initial_facility_operator(current, mode, rank_index, area_cache)
                if result is None:
                    continue
                candidate_cost = result['candidate_total']
                if candidate_cost >= current_cost - CLRP_INITIAL_FACILITY_MIN_IMPROVEMENT:
                    continue
                old_cost = current_cost
                current = result['routes']
                current_cost = candidate_cost
                accepted_moves += 1
                improved_this_pass = True
                area_cache.clear()
                record = {
                    'pass': pass_index,
                    'mode': mode,
                    'old_group': list(result['old_group']),
                    'replacement': list(result['replacement']),
                    'old_cost': float(old_cost),
                    'new_cost': float(current_cost),
                    'improvement': float(old_cost - current_cost),
                    'improvement_ratio': float((old_cost - current_cost) / max(abs(old_cost), 1e-12)),
                    'open_depots': len(current),
                    'local_point_count': result['point_count'],
                    'local_coverage_area_m2': result['coverage_area'],
                }
                self.initial_facility_refinement_history.append(record)
                improvement_pct = 100.0 * (old_cost - current_cost) / max(abs(old_cost), 1e-12)
                print(f"[initial depot refine] {mode_labels[mode]}: {result['old_group']} → {result['replacement']}; "
                      f"Cost {old_cost:.2f} → {current_cost:.2f}, reduction {old_cost - current_cost:.2f} "
                      f"({improvement_pct:.2f}%); open depots={len(current)}")
            if improved_this_pass:
                no_improve_passes = 0
            else:
                no_improve_passes += 1
                print(f'[initial depot refine] pass {pass_index} with no improvement, consecutive={no_improve_passes}/'
                      f'{CLRP_INITIAL_FACILITY_NO_IMPROVE_LIMIT}')
        feasible, violations = self.validate_solution(current)
        if not feasible:
            raise ValueError('initial depot-structure refinement produced an infeasible solution: ' + '; '.join(violations[:8]))
        print(f'initial depot-structure refinement done: accepted {accepted_moves} move(s); open depots {initial_depots} → {len(current)}; '
              f'total cost {initial_cost:.2f} → {current_cost:.2f}, reduction {initial_cost - current_cost:.2f}.')
        return current

    def initial_solution(self):
        print('\n=== Building visibility-greedy initial solution ===')
        selected = self._construct_visibility_greedy_depots()
        print(f'visibility-greedy selected depots: {sorted(selected)} ({len(selected)})')
        routes, failure = self._construct_visibility_greedy_routes(selected)
        if routes is None:
            raise ValueError(f'visibility-greedy initial solution failed: {failure}')
        feasible, violations = self.validate_solution(routes)
        if not feasible:
            raise ValueError('visibility-greedy initial solution violates hard constraints: ' + '; '.join(violations[:8]))
        routes = self._refine_initial_facility_structure(routes)
        self.selected_depots = list(routes)
        self.best_build_cost = self.calculate_build_cost(self.selected_depots)
        self.best_transport_cost = self.calculate_total_transport_cost(routes)
        self.best_total_cost = self.best_build_cost + self.best_transport_cost
        self.best_routes = deepcopy(routes)
        print('Visibility-greedy routes after depot-structure refinement become the ALNS initial solution.')
        print(f'initial construction cost: {self.best_build_cost:.2f}')
        print(f'initial transport cost: {self.best_transport_cost:.2f}')
        print(f'initial total cost: {self.best_total_cost:.2f}')
        return routes

    def _two_opt_route(self, route):
        if len(route) <= 4:
            return route
        improved = True
        best = list(route)
        while improved:
            improved = False
            for i in range(1, len(best) - 3):
                for j in range(i + 2, len(best) - 1):
                    old_cost = self.dist_matrix[best[i - 1]][best[i]] + self.dist_matrix[best[j]][best[j + 1]]
                    new_cost = self.dist_matrix[best[i - 1]][best[j]] + self.dist_matrix[best[i]][best[j + 1]]
                    if self.edge_crosses_no_fly(best[i - 1], best[j]):
                        continue
                    if self.edge_crosses_no_fly(best[i], best[j + 1]):
                        continue
                    if new_cost < old_cost - 1e-10:
                        candidate = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                        if self.check_route_length(candidate):
                            best = candidate
                            improved = True
        return best

class AdaptiveALNS_CLRP(ALNS_CLRP):

    def __init__(self, *args, reaction_factor=0.2, segment_length=CLRP_SEGMENT_LENGTH, local_search_interval=CLRP_LOCAL_SEARCH_INTERVAL, elite_size=CLRP_ELITE_SIZE, stagnation_limit=CLRP_STAGNATION_LIMIT, min_destroy_rate=CLRP_MIN_DESTROY_RATE, max_destroy_rate=CLRP_MAX_DESTROY_RATE, max_destroy_customers=CLRP_MAX_DESTROY_CUSTOMERS, **kwargs):
        super().__init__(*args, **kwargs)
        self.reaction_factor = float(reaction_factor)
        self.segment_length = max(1, int(segment_length))
        self.local_search_interval = max(1, int(local_search_interval))
        self.elite_size = max(1, int(elite_size))
        self.stagnation_limit = max(1, int(stagnation_limit))
        self.min_destroy_rate = float(min_destroy_rate)
        self.max_destroy_rate = float(max_destroy_rate)
        self.max_destroy_customers = None if max_destroy_customers is None else max(1, int(max_destroy_customers))
        if not 0.0 < self.min_destroy_rate <= self.max_destroy_rate <= 1.0:
            raise ValueError('destroy rates must satisfy 0 < min_destroy_rate <= max_destroy_rate <= 1')
        self.destroy_operators = {
            'random': self._destroy_random,
            'worst': self._destroy_worst,
            'related': self._destroy_related,
            'route': self._destroy_route,
            'cluster': self._destroy_cluster,
            'facility_relocate_2_to_2': self._destroy_facility_relocate_2_to_2,
            'facility_reduce_3_to_2': self._destroy_facility_reduce_3_to_2,
            'facility_expand_3_to_4': self._destroy_facility_expand_3_to_4,
        }
        if _ablation_disable_alns_facility_ops():
            self.destroy_operators = {
                name: fn for name, fn in self.destroy_operators.items()
                if not name.startswith('facility_')
            }
        self.repair_operators = {'greedy': self._repair_greedy, 'regret2': lambda r, c: self._repair_regret_k(r, c, 2), 'regret3': lambda r, c: self._repair_regret_k(r, c, 3), 'regret4': lambda r, c: self._repair_regret_k(r, c, 4)}
        self.destroy_weights = {
            name: (CLRP_ALNS_FACILITY_INITIAL_WEIGHT if name.startswith('facility_') else 1.0)
            for name in self.destroy_operators
        }
        self.repair_weights = {name: 1.0 for name in self.repair_operators}
        self.destroy_scores = {name: 0.0 for name in self.destroy_operators}
        self.repair_scores = {name: 0.0 for name in self.repair_operators}
        self.destroy_uses = {name: 0 for name in self.destroy_operators}
        self.repair_uses = {name: 0 for name in self.repair_operators}
        self.elite_pool = []

    def _route_load(self, route):
        return float(sum((self.demands[n] for n in route[1:-1])))

    def _route_is_feasible(self, route, depot=None):
        if depot is None:
            depot = route[0]
        if len(route) < 3 or route[0] != depot or route[-1] != depot:
            return False
        if self._route_load(route) > self.vehicle_cap + 1e-10:
            return False
        if not self.check_route_length(route):
            return False
        if not self.check_route_visibility(route, depot):
            return False
        return all((not self.edge_crosses_no_fly(a, b) for a, b in zip(route[:-1], route[1:])))

    def _solution_signature(self, routes_dict):
        depot_sig = tuple(sorted(routes_dict))
        route_sig = []
        for depot in depot_sig:
            normalized = [tuple(route[1:-1]) for route in routes_dict[depot]]
            route_sig.append((depot, tuple(sorted(normalized))))
        return (depot_sig, tuple(route_sig))

    def _push_elite(self, routes_dict, total_cost):
        sig = self._solution_signature(routes_dict)
        for _, old_sig, _ in self.elite_pool:
            if old_sig == sig:
                return
        self.elite_pool.append((float(total_cost), sig, deepcopy(routes_dict)))
        self.elite_pool.sort(key=lambda item: item[0])
        self.elite_pool = self.elite_pool[:self.elite_size]

    def _roulette_select(self, weights):
        names = list(weights)
        values = np.asarray([max(weights[n], 1e-12) for n in names], dtype=float)
        values /= values.sum()
        return str(np.random.choice(names, p=values))

    def _reset_segment_statistics(self):
        for name in self.destroy_scores:
            self.destroy_scores[name] = 0.0
            self.destroy_uses[name] = 0
        for name in self.repair_scores:
            self.repair_scores[name] = 0.0
            self.repair_uses[name] = 0

    def _update_operator_weights(self):
        rho = self.reaction_factor
        for name in self.destroy_weights:
            if self.destroy_uses[name] > 0:
                avg_score = self.destroy_scores[name] / self.destroy_uses[name]
                self.destroy_weights[name] = max(0.05, (1.0 - rho) * self.destroy_weights[name] + rho * avg_score)
        for name in self.repair_weights:
            if self.repair_uses[name] > 0:
                avg_score = self.repair_scores[name] / self.repair_uses[name]
                self.repair_weights[name] = max(0.05, (1.0 - rho) * self.repair_weights[name] + rho * avg_score)
        self._reset_segment_statistics()

    def _remove_customers(self, routes_dict, customers_to_remove):
        remove_set = set(customers_to_remove)
        destroyed = deepcopy(routes_dict)
        removed = []
        for depot in list(destroyed):
            new_routes = []
            for route in destroyed[depot]:
                kept = [depot]
                for node in route[1:-1]:
                    if node in remove_set:
                        removed.append(node)
                    else:
                        kept.append(node)
                kept.append(depot)
                if len(kept) > 2:
                    new_routes.append(kept)
            destroyed[depot] = new_routes
        return (destroyed, list(dict.fromkeys(removed)))

    def _all_customer_records(self, routes_dict):
        records = []
        for depot, routes in routes_dict.items():
            for ri, route in enumerate(routes):
                for ci in range(1, len(route) - 1):
                    records.append((depot, ri, ci, route[ci]))
        return records

    def _destroy_count(self, routes_dict, destroy_rate):
        n = len(self._all_customer_records(routes_dict))
        q = min(n, max(1, int(round(n * destroy_rate))))
        if self.max_destroy_customers is not None:
            q = min(q, self.max_destroy_customers)
        return q

    def _destroy_random(self, routes_dict, destroy_rate):
        records = self._all_customer_records(routes_dict)
        q = min(len(records), self._destroy_count(routes_dict, destroy_rate))
        chosen = random.sample([r[3] for r in records], q)
        return self._remove_customers(routes_dict, chosen)

    def _destroy_worst(self, routes_dict, destroy_rate):
        records = self._all_customer_records(routes_dict)
        scored = []
        for depot, ri, ci, cust in records:
            route = routes_dict[depot][ri]
            prev_n, next_n = (route[ci - 1], route[ci + 1])
            saving = self.dist_matrix[prev_n][cust] + self.dist_matrix[cust][next_n] - self.dist_matrix[prev_n][next_n]
            scored.append((saving * random.uniform(0.85, 1.15), cust))
        scored.sort(reverse=True)
        q = min(len(scored), self._destroy_count(routes_dict, destroy_rate))
        return self._remove_customers(routes_dict, [cust for _, cust in scored[:q]])

    def _destroy_related(self, routes_dict, destroy_rate):
        records = self._all_customer_records(routes_dict)
        if not records:
            return (deepcopy(routes_dict), [])
        q = min(len(records), self._destroy_count(routes_dict, destroy_rate))
        seed = random.choice(records)
        seed_depot, seed_ri, _, seed_cust = seed
        finite_distances = self.actual_dist[np.isfinite(self.actual_dist)]
        dist_scale = max(float(np.max(finite_distances)) if finite_distances.size else 1.0, 1.0)
        demand_scale = max(float(np.max(self.demands)), 1.0)
        related = []
        for depot, ri, _, cust in records:
            score = self.actual_dist[seed_cust][cust] / dist_scale + 0.35 * abs(self.demands[seed_cust] - self.demands[cust]) / demand_scale + (0.0 if depot == seed_depot else 0.35) + (0.0 if depot == seed_depot and ri == seed_ri else 0.15)
            related.append((score * random.uniform(0.9, 1.1), cust))
        related.sort()
        return self._remove_customers(routes_dict, [cust for _, cust in related[:q]])

    def _destroy_route(self, routes_dict, destroy_rate):
        q = self._destroy_count(routes_dict, destroy_rate)
        route_records = []
        for depot, routes in routes_dict.items():
            for ri, route in enumerate(routes):
                customers = route[1:-1]
                if customers:
                    score = self.calculate_route_cost(route) / len(customers)
                    route_records.append((score * random.uniform(0.9, 1.1), depot, ri, list(customers)))
        route_records.sort(reverse=True)
        removed = []
        for _, _, _, customers in route_records:
            removed.extend(customers)
            if len(set(removed)) >= q:
                break
        return self._remove_customers(routes_dict, list(dict.fromkeys(removed))[:q])

    def _destroy_cluster(self, routes_dict, destroy_rate):
        records = self._all_customer_records(routes_dict)
        if not records:
            return (deepcopy(routes_dict), [])
        q = min(len(records), self._destroy_count(routes_dict, destroy_rate))
        seed = random.choice(records)[3]
        ordered = sorted(((self.actual_dist[seed][rec[3]] * random.uniform(0.95, 1.05), rec[3]) for rec in records))
        return self._remove_customers(routes_dict, [cust for _, cust in ordered[:q]])

    def _destroy_facility_structure(self, routes_dict, mode):
        """Produce an ALNS candidate that truly changes depot structure via a full local rebuild.

        These operators no longer do the old close-one/open-one swap; a candidate is returned only if
        the current block cost falls by more than CLRP_ALNS_FACILITY_MIN_REL_IMPROVEMENT.
        """
        area_cache = {}
        ranked_groups = self._rank_initial_facility_groups(routes_dict, mode, area_cache)
        if not ranked_groups:
            return (None, None)
        rank_pool = min(len(ranked_groups), max(1, int(CLRP_ALNS_FACILITY_RANK_POOL)))
        rank_index = random.randrange(rank_pool)
        result = self._try_initial_facility_operator(
            routes_dict,
            mode,
            rank_index,
            area_cache,
            min_relative_improvement=CLRP_ALNS_FACILITY_MIN_REL_IMPROVEMENT,
            rank_limit=rank_pool)
        if result is None:
            return (None, None)
        self._last_facility_move = {
            'mode': mode,
            'old_group': list(result['old_group']),
            'replacement': list(result['replacement']),
            'improvement_ratio': float(result['improvement_ratio']),
        }
        # The candidate is already complete and globally feasible; when removed=[] the repair operator only validates.
        return (deepcopy(result['routes']), [])

    def _destroy_facility_relocate_2_to_2(self, routes_dict, destroy_rate):
        return self._destroy_facility_structure(routes_dict, 'relocate_2_to_2')

    def _destroy_facility_reduce_3_to_2(self, routes_dict, destroy_rate):
        return self._destroy_facility_structure(routes_dict, 'reduce_3_to_2')

    def _destroy_facility_expand_3_to_4(self, routes_dict, destroy_rate):
        return self._destroy_facility_structure(routes_dict, 'expand_3_to_4')

    def _prepare_repair(self, destroyed_routes, removed_customers):
        repaired = deepcopy(destroyed_routes)
        pending = list(dict.fromkeys(removed_customers))
        return (repaired, pending)

    def _feasible_insertions(self, routes_dict, cust, noisy=False):
        options = []
        for depot, routes in routes_dict.items():
            if not self.can_depot_serve_customer(depot, cust):
                continue
            for route_idx, route in enumerate(routes):
                if self._route_load(route) + self.demands[cust] > self.vehicle_cap + 1e-10:
                    continue
                old_actual = self.calculate_route_length_actual(route)
                for pos in range(1, len(route)):
                    a, b = (route[pos - 1], route[pos])
                    if self.edge_crosses_no_fly(a, cust) or self.edge_crosses_no_fly(cust, b):
                        continue
                    actual_inc = self.actual_dist[a][cust] + self.actual_dist[cust][b] - self.actual_dist[a][b]
                    if old_actual + actual_inc > self.max_route_length + 1e-10:
                        continue
                    delta = self.dist_matrix[a][cust] + self.dist_matrix[cust][b] - self.dist_matrix[a][b]
                    ranking_delta = delta
                    if noisy:
                        ranking_delta += random.uniform(-0.05, 0.05) * max(abs(delta), 1.0)
                    options.append((ranking_delta, delta, depot, route_idx, pos, False))
            if len(routes) < self.max_veh_per_depot and self.demands[cust] <= self.vehicle_cap and self.can_depot_serve_customer(depot, cust) and (not self.edge_crosses_no_fly(depot, cust)) and (2.0 * self.actual_dist[depot][cust] <= self.max_route_length):
                delta = 2.0 * self.dist_matrix[depot][cust]
                ranking_delta = delta
                if noisy:
                    ranking_delta += random.uniform(-0.05, 0.05) * max(abs(delta), 1.0)
                options.append((ranking_delta, delta, depot, -1, -1, True))
        options.sort(key=lambda item: item[0])
        return options

    @staticmethod
    def _apply_insertion(routes_dict, cust, option):
        _, _, depot, route_idx, pos, is_new = option
        if is_new:
            routes_dict[depot].append([depot, cust, depot])
        else:
            routes_dict[depot][route_idx].insert(pos, cust)

    def _repair_greedy(self, destroyed_routes, removed_customers):
        repaired, pending = self._prepare_repair(destroyed_routes, removed_customers)
        if repaired is None:
            return None
        pending = list(pending)
        while pending:
            best = None
            for cust in pending:
                options = self._feasible_insertions(repaired, cust, noisy=True)
                if options:
                    candidate = (options[0][0], -len(options), cust, options[0])
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                return None
            _, _, cust, option = best
            self._apply_insertion(repaired, cust, option)
            pending.remove(cust)
        feasible, _ = self.validate_solution(repaired)
        return repaired if feasible else None

    def _repair_regret_k(self, destroyed_routes, removed_customers, k):
        repaired, pending = self._prepare_repair(destroyed_routes, removed_customers)
        if repaired is None:
            return None
        pending = list(pending)
        while pending:
            choice = None
            for cust in pending:
                options = self._feasible_insertions(repaired, cust, noisy=False)
                if not options:
                    continue
                best_cost = options[0][1]
                considered = options[:k]
                if len(considered) < k:
                    regret = 1000000000000000.0 + (k - len(considered)) * 1000000000000.0 - best_cost
                else:
                    regret = sum((option[1] - best_cost for option in considered[1:]))
                candidate = (regret, -best_cost, -len(options), cust, options[0])
                if choice is None or candidate > choice:
                    choice = candidate
            if choice is None:
                return None
            _, _, _, cust, option = choice
            self._apply_insertion(repaired, cust, option)
            pending.remove(cust)
        feasible, _ = self.validate_solution(repaired)
        return repaired if feasible else None

    def repair_solution(self, destroyed_routes, removed_customers):
        return self._repair_regret_k(destroyed_routes, removed_customers, 3)

    def _or_opt_route(self, route, max_segment=3):
        best = list(route)
        improved = True
        while improved:
            improved = False
            old_cost = self.calculate_route_cost(best)
            n_customers = len(best) - 2
            for seg_len in range(1, min(max_segment, n_customers) + 1):
                for start in range(1, len(best) - seg_len):
                    segment = best[start:start + seg_len]
                    remaining = best[:start] + best[start + seg_len:]
                    for pos in range(1, len(remaining)):
                        candidate = remaining[:pos] + segment + remaining[pos:]
                        if candidate == best or not self._route_is_feasible(candidate):
                            continue
                        new_cost = self.calculate_route_cost(candidate)
                        if new_cost < old_cost - 1e-10:
                            best = candidate
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best

    def _improve_intra_routes(self, routes_dict):
        changed = False
        for depot in routes_dict:
            for ri, route in enumerate(routes_dict[depot]):
                old_cost = self.calculate_route_cost(route)
                candidate = self._two_opt_route(route)
                candidate = self._or_opt_route(candidate)
                if self.calculate_route_cost(candidate) < old_cost - 1e-10:
                    routes_dict[depot][ri] = candidate
                    changed = True
        return changed

    def _first_relocate(self, routes_dict):
        depots = list(routes_dict)
        for src_depot in depots:
            for src_ri, src_route in enumerate(list(routes_dict[src_depot])):
                for src_pos in range(1, len(src_route) - 1):
                    cust = src_route[src_pos]
                    new_src = src_route[:src_pos] + src_route[src_pos + 1:]
                    src_delta = -self.calculate_route_cost(src_route)
                    if len(new_src) > 2:
                        if not self._route_is_feasible(new_src, src_depot):
                            continue
                        src_delta += self.calculate_route_cost(new_src)
                    for dst_depot in depots:
                        for dst_ri, dst_route in enumerate(routes_dict[dst_depot]):
                            if src_depot == dst_depot and src_ri == dst_ri:
                                continue
                            if self._route_load(dst_route) + self.demands[cust] > self.vehicle_cap:
                                continue
                            for pos in range(1, len(dst_route)):
                                new_dst = dst_route[:pos] + [cust] + dst_route[pos:]
                                if not self._route_is_feasible(new_dst, dst_depot):
                                    continue
                                delta = src_delta + self.calculate_route_cost(new_dst) - self.calculate_route_cost(dst_route)
                                if delta < -1e-10:
                                    routes_dict[src_depot][src_ri] = new_src
                                    routes_dict[dst_depot][dst_ri] = new_dst
                                    if len(new_src) <= 2:
                                        routes_dict[src_depot].pop(src_ri)
                                    return True
                        if len(routes_dict[dst_depot]) < self.max_veh_per_depot and (not (src_depot == dst_depot and len(new_src) <= 2)):
                            new_dst = [dst_depot, cust, dst_depot]
                            if self._route_is_feasible(new_dst, dst_depot):
                                delta = src_delta + self.calculate_route_cost(new_dst)
                                if delta < -1e-10:
                                    routes_dict[src_depot][src_ri] = new_src
                                    routes_dict[dst_depot].append(new_dst)
                                    if len(new_src) <= 2:
                                        routes_dict[src_depot].pop(src_ri)
                                    return True
        return False

    def _first_swap(self, routes_dict):
        refs = [(d, ri) for d in routes_dict for ri in range(len(routes_dict[d]))]
        for idx_a, (da, ria) in enumerate(refs):
            ra = routes_dict[da][ria]
            for db, rib in refs[idx_a + 1:]:
                rb = routes_dict[db][rib]
                old_cost = self.calculate_route_cost(ra) + self.calculate_route_cost(rb)
                for ia in range(1, len(ra) - 1):
                    ca = ra[ia]
                    for ib in range(1, len(rb) - 1):
                        cb = rb[ib]
                        new_ra = list(ra)
                        new_rb = list(rb)
                        new_ra[ia], new_rb[ib] = (cb, ca)
                        if not self._route_is_feasible(new_ra, da) or not self._route_is_feasible(new_rb, db):
                            continue
                        new_cost = self.calculate_route_cost(new_ra) + self.calculate_route_cost(new_rb)
                        if new_cost < old_cost - 1e-10:
                            routes_dict[da][ria] = new_ra
                            routes_dict[db][rib] = new_rb
                            return True
        return False

    def _first_two_opt_star(self, routes_dict):
        refs = [(d, ri) for d in routes_dict for ri in range(len(routes_dict[d]))]
        for idx_a, (da, ria) in enumerate(refs):
            ra = routes_dict[da][ria]
            for db, rib in refs[idx_a + 1:]:
                rb = routes_dict[db][rib]
                old_cost = self.calculate_route_cost(ra) + self.calculate_route_cost(rb)
                for cut_a in range(0, len(ra) - 2):
                    for cut_b in range(0, len(rb) - 2):
                        prefix_a, tail_a = (ra[1:cut_a + 2], ra[cut_a + 2:-1])
                        prefix_b, tail_b = (rb[1:cut_b + 2], rb[cut_b + 2:-1])
                        new_ra = [da] + prefix_a + tail_b + [da]
                        new_rb = [db] + prefix_b + tail_a + [db]
                        if len(new_ra) <= 2 or len(new_rb) <= 2:
                            continue
                        if not self._route_is_feasible(new_ra, da) or not self._route_is_feasible(new_rb, db):
                            continue
                        new_cost = self.calculate_route_cost(new_ra) + self.calculate_route_cost(new_rb)
                        if new_cost < old_cost - 1e-10:
                            routes_dict[da][ria] = new_ra
                            routes_dict[db][rib] = new_rb
                            return True
        return False

    def _local_search_intra_routes(self, routes_dict, max_rounds=4):
        # max_rounds=0 is a strict VND ablation: no route-level operators or terminal polish.
        if int(max_rounds) <= 0:
            return deepcopy(routes_dict)
        routes = deepcopy(routes_dict)
        for _ in range(max_rounds):
            changed = self._improve_intra_routes(routes)
            if self._first_relocate(routes):
                changed = True
                continue
            if self._first_swap(routes):
                changed = True
                continue
            if self._first_two_opt_star(routes):
                changed = True
                continue
            if not changed:
                break
        for depot in routes:
            routes[depot] = [r for r in routes[depot] if len(r) > 2]
        feasible, _ = self.validate_solution(routes)
        return routes if feasible else deepcopy(routes_dict)

    def _vnd_rounds_for_candidate(self, candidate_cost, current_cost, iteration):
        """Allocate VND budget by candidate potential; with adaptive mode off, reproduce the old trigger rule exactly."""
        fixed_rounds = max(0, int(globals().get('CLRP_VND_MAX_ROUNDS', 4)))
        if fixed_rounds <= 0:
            return 0
        if not bool(globals().get('CLRP_VND_ADAPTIVE_MODE', False)):
            return fixed_rounds if (candidate_cost < current_cost or iteration % self.local_search_interval == 0) else 0
        light = max(0, int(globals().get('CLRP_VND_LIGHT_MAX_ROUNDS', 1)))
        elite = max(light, int(globals().get('CLRP_VND_ELITE_MAX_ROUNDS', fixed_rounds)))
        gap = max(0.0, float(globals().get('CLRP_VND_ELITE_GAP_RATIO', 0.003)))
        periodic = max(1, int(globals().get('CLRP_VND_PERIODIC_INTERVAL', 100)))
        # Candidates within the gap of the historic best deserve a deep polish; strict new bests are included.
        if candidate_cost <= self.best_total_cost * (1.0 + gap):
            return elite
        if candidate_cost < current_cost or iteration % periodic == 0:
            return light
        return 0


def _write_pickle_atomic(path, payload):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as file_obj:
        pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    return path


def _prepare_resume_history(path, fields, completed_iteration):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or completed_iteration <= 0:
        return 'w'
    try:
        with open(path, 'r', newline='', encoding='utf-8-sig') as file_obj:
            rows = list(csv.DictReader(file_obj))
        kept = []
        for row in rows:
            try:
                iteration = int(row.get('iteration', -1))
            except (TypeError, ValueError):
                continue
            if iteration <= completed_iteration and row.get('phase') != 'final_vnd':
                kept.append(row)
        with open(path, 'w', newline='', encoding='utf-8-sig') as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)
        return 'a'
    except Exception:
        return 'a'


def _is_significant_best_improvement(candidate_cost, best_cost, min_rel=None):
    """Whether the candidate improves the historic best by more than the minimum relative threshold (default 0.01%)."""
    threshold = CLRP_BEST_MIN_REL_IMPROVEMENT if min_rel is None else float(min_rel)
    best = float(best_cost)
    cand = float(candidate_cost)
    if not math.isfinite(best) or not math.isfinite(cand):
        return False
    if best <= 0.0:
        return cand < best - 1e-10
    return (best - cand) / best > threshold


class ExperimentALNS_CLRP(AdaptiveALNS_CLRP):
    HISTORY_FIELDS = ['scenario', 'seed', 'iteration', 'phase', 'current_total_cost', 'candidate_total_cost', 'best_total_cost', 'best_build_cost', 'best_transport_cost', 'temperature', 'destroy_rate', 'destroy_operator', 'repair_operator', 'candidate_feasible', 'accepted', 'global_best_updated', 'reward', 'no_improve_iterations', 'n_open_depots', 'n_routes', 'selected_depots', 'elapsed_seconds', 'destroy_weights', 'repair_weights']

    @staticmethod
    def _route_count(routes):
        return sum((len(route_list) for route_list in routes.values()))

    def _history_row(self, *, scenario, seed, iteration, phase, current_cost, candidate_cost, temperature, destroy_rate, destroy_operator, repair_operator, candidate_feasible, accepted, global_best_updated, reward, no_improve, elapsed_seconds):
        return {'scenario': scenario, 'seed': int(seed), 'iteration': int(iteration), 'phase': phase, 'current_total_cost': f'{current_cost:.10f}', 'candidate_total_cost': '' if candidate_cost is None else f'{candidate_cost:.10f}', 'best_total_cost': f'{self.best_total_cost:.10f}', 'best_build_cost': f'{self.best_build_cost:.10f}', 'best_transport_cost': f'{self.best_transport_cost:.10f}', 'temperature': f'{temperature:.10f}', 'destroy_rate': '' if destroy_rate is None else f'{destroy_rate:.10f}', 'destroy_operator': destroy_operator or '', 'repair_operator': repair_operator or '', 'candidate_feasible': int(bool(candidate_feasible)), 'accepted': int(bool(accepted)), 'global_best_updated': int(bool(global_best_updated)), 'reward': f'{float(reward):.6f}', 'no_improve_iterations': int(no_improve), 'n_open_depots': len(self.selected_depots), 'n_routes': self._route_count(self.best_routes), 'selected_depots': ';'.join(map(str, sorted(self.selected_depots))), 'elapsed_seconds': f'{elapsed_seconds:.6f}', 'destroy_weights': json.dumps(self.destroy_weights, ensure_ascii=False, sort_keys=True), 'repair_weights': json.dumps(self.repair_weights, ensure_ascii=False, sort_keys=True)}

    def alns_optimize(self, max_iter=CLRP_MAX_ITER, destroy_rate=CLRP_DESTROY_RATE, T_start=None, T_end=None, cooling_rate=None, start_worse_ratio=CLRP_T_START_WORSE_RATIO, start_accept_prob=CLRP_T_START_ACCEPT_PROB, end_cost_ratio=CLRP_T_END_COST_RATIO, history_csv_path=None, checkpoint_path=None, random_seed=CLRP_RANDOM_SEED, scenario_label='unspecified', print_interval=CLRP_CONSOLE_PROGRESS_INTERVAL, progress_callback=None):
        prefix = f'[{scenario_label} | seed={int(random_seed)}]'
        print(f'\n{prefix} === Starting enhanced adaptive ALNS for CLRP ===')
        run_start = time.time()
        elapsed_before = 0.0
        max_iter = max(1, int(max_iter))
        destroy_rate = float(destroy_rate)
        print_interval = max(0, int(print_interval))
        if not 0.0 < destroy_rate <= 1.0:
            raise ValueError('destroy_rate must lie in (0, 1]')
        checkpoint_path = os.path.abspath(os.fspath(checkpoint_path)) if checkpoint_path else None
        run_signature = {
            'max_destroy_customers': CLRP_MAX_DESTROY_CUSTOMERS,
            'local_search_interval': CLRP_LOCAL_SEARCH_INTERVAL,
            'facility_improvement': globals().get('CLRP_ALNS_FACILITY_MIN_REL_IMPROVEMENT'),
            'initial_expand_improvement': globals().get('CLRP_INITIAL_FACILITY_EXPAND_MIN_REL_IMPROVEMENT'),
        }
        checkpoint = None
        if CLRP_ALNS_RESUME and checkpoint_path and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'rb') as file_obj:
                    candidate_state = pickle.load(file_obj)
                compatible = (
                    candidate_state.get('format_version') == 1
                    and candidate_state.get('scenario') == str(scenario_label)
                    and int(candidate_state.get('seed', -1)) == int(random_seed)
                    and int(candidate_state.get('max_iter', -1)) == max_iter
                    and candidate_state.get('run_signature') == run_signature
                    and set(candidate_state.get('destroy_weights', {})) == set(self.destroy_weights)
                    and set(candidate_state.get('repair_weights', {})) == set(self.repair_weights)
                )
                if compatible:
                    checkpoint = candidate_state
                else:
                    print(f'{prefix} checkpoint is incompatible with this job, ignoring: {checkpoint_path}')
            except Exception as exc:
                print(f'{prefix} checkpoint read failed, restarting: {exc}')

        if checkpoint is None:
            current_routes = self.initial_solution()
            current_cost = self.calculate_build_cost(current_routes.keys()) + self.calculate_total_transport_cost(current_routes)
            self.initial_total_cost = current_cost
            self._push_elite(current_routes, current_cost)
            if T_start is None:
                if not 0.0 < start_accept_prob < 1.0:
                    raise ValueError('start_accept_prob must lie in (0, 1)')
                if start_worse_ratio <= 0.0:
                    raise ValueError('start_worse_ratio must be greater than 0')
                T_start = max(1e-09, -(float(start_worse_ratio) * current_cost) / math.log(float(start_accept_prob)))
            if T_end is None:
                if end_cost_ratio <= 0.0:
                    raise ValueError('end_cost_ratio must be greater than 0')
                T_end = max(1e-09, float(end_cost_ratio) * current_cost)
            T_start, T_end = float(T_start), float(T_end)
            if T_start <= 0.0 or T_end <= 0.0 or T_end >= T_start:
                raise ValueError('temperature must satisfy T_start > T_end > 0')
            if cooling_rate is None:
                cooling_rate = (T_end / T_start) ** (1.0 / max_iter)
            cooling_rate = float(cooling_rate)
            if not 0.0 < cooling_rate < 1.0:
                raise ValueError('cooling_rate must lie in (0, 1)')
            temperature = T_start
            no_improve = 0
            self._reset_segment_statistics()
            start_iteration = 1
            last_completed_iteration = 0
        else:
            current_routes = checkpoint['current_routes']
            current_cost = float(checkpoint['current_cost'])
            self.initial_total_cost = float(checkpoint['initial_total_cost'])
            self.best_routes = checkpoint['best_routes']
            self.selected_depots = set(checkpoint['selected_depots'])
            self.best_build_cost = float(checkpoint['best_build_cost'])
            self.best_transport_cost = float(checkpoint['best_transport_cost'])
            self.best_total_cost = float(checkpoint['best_total_cost'])
            self.destroy_weights = checkpoint['destroy_weights']
            self.repair_weights = checkpoint['repair_weights']
            self.destroy_scores = checkpoint['destroy_scores']
            self.repair_scores = checkpoint['repair_scores']
            self.destroy_uses = checkpoint['destroy_uses']
            self.repair_uses = checkpoint['repair_uses']
            self.elite_pool = checkpoint['elite_pool']
            self.initial_facility_refinement_history = checkpoint.get('initial_facility_refinement_history', [])
            T_start = float(checkpoint['T_start'])
            T_end = float(checkpoint['T_end'])
            cooling_rate = float(checkpoint['cooling_rate'])
            temperature = float(checkpoint['temperature'])
            no_improve = int(checkpoint['no_improve'])
            last_completed_iteration = int(checkpoint['iteration'])
            start_iteration = last_completed_iteration + 1
            elapsed_before = float(checkpoint.get('elapsed_seconds', 0.0))
            random.setstate(checkpoint['random_state'])
            np.random.set_state(checkpoint['numpy_random_state'])
            print(f'{prefix} restored checkpoint: finished {last_completed_iteration}/{max_iter} iterations, Best={self.best_total_cost:.2f}')
            if checkpoint.get('completed'):
                feasible, violations = self.validate_solution(self.best_routes)
                if not feasible:
                    raise RuntimeError('completed checkpoint failed constraint validation: ' + '; '.join(violations))
                self.history_csv_path = history_csv_path
                return (self.selected_depots, self.best_routes, self.best_build_cost, self.best_transport_cost, self.best_total_cost, elapsed_before)

        def elapsed_total():
            return elapsed_before + time.time() - run_start

        def save_checkpoint(completed=False):
            if not (CLRP_ALNS_RESUME and checkpoint_path):
                return
            state = {
                'format_version': 1, 'scenario': str(scenario_label), 'seed': int(random_seed),
                'max_iter': max_iter, 'run_signature': run_signature, 'iteration': int(last_completed_iteration), 'completed': bool(completed),
                'current_routes': current_routes, 'current_cost': float(current_cost),
                'initial_total_cost': float(self.initial_total_cost), 'best_routes': self.best_routes,
                'selected_depots': sorted(self.selected_depots), 'best_build_cost': float(self.best_build_cost),
                'best_transport_cost': float(self.best_transport_cost), 'best_total_cost': float(self.best_total_cost),
                'T_start': float(T_start), 'T_end': float(T_end), 'cooling_rate': float(cooling_rate),
                'temperature': float(temperature), 'no_improve': int(no_improve),
                'destroy_weights': self.destroy_weights, 'repair_weights': self.repair_weights,
                'destroy_scores': self.destroy_scores, 'repair_scores': self.repair_scores,
                'destroy_uses': self.destroy_uses, 'repair_uses': self.repair_uses,
                'elite_pool': self.elite_pool,
                'initial_facility_refinement_history': getattr(self, 'initial_facility_refinement_history', []),
                'random_state': random.getstate(), 'numpy_random_state': np.random.get_state(),
                'elapsed_seconds': float(elapsed_total()),
            }
            _write_pickle_atomic(checkpoint_path, state)

        history_handle = history_writer = None
        if history_csv_path:
            history_csv_path = os.path.abspath(os.fspath(history_csv_path))
            mode = _prepare_resume_history(history_csv_path, self.HISTORY_FIELDS, last_completed_iteration) if checkpoint else 'w'
            history_handle = open(history_csv_path, mode, newline='', encoding='utf-8-sig')
            history_writer = csv.DictWriter(history_handle, fieldnames=self.HISTORY_FIELDS)
            if mode == 'w':
                history_writer.writeheader()
                history_writer.writerow(self._history_row(scenario=scenario_label, seed=random_seed, iteration=0, phase='initial', current_cost=current_cost, candidate_cost=None, temperature=temperature, destroy_rate=None, destroy_operator=None, repair_operator=None, candidate_feasible=True, accepted=True, global_best_updated=False, reward=0.0, no_improve=no_improve, elapsed_seconds=elapsed_total()))
                history_handle.flush()
        print(f'{prefix} max iterations: {max_iter}')
        print(f'{prefix} start temperature: {T_start:.6f}, end temperature: {T_end:.6f}, cooling rate: {cooling_rate:.8f}')
        print(f"{prefix} resume file: {checkpoint_path or 'disabled'}")
        perf = {'destroy': 0.0, 'repair': 0.0, 'vnd': 0.0, 'cost_and_accept': 0.0, 'history_io': 0.0}
        flush_interval = max(1, int(globals().get('CLRP_HISTORY_FLUSH_INTERVAL', CLRP_ALNS_CHECKPOINT_INTERVAL)))
        save_checkpoint(False)
        try:
            for iteration in range(start_iteration, max_iter + 1):
                destroy_name = self._roulette_select(self.destroy_weights)
                repair_name = self._roulette_select(self.repair_weights)
                self.destroy_uses[destroy_name] += 1
                self.repair_uses[repair_name] += 1
                stagnation_span = max(self.stagnation_limit, 1)
                stagnation_factor = min(self.max_destroy_rate - destroy_rate, no_improve / stagnation_span * 0.08)
                adaptive_rate = min(self.max_destroy_rate, max(self.min_destroy_rate, destroy_rate + stagnation_factor + random.uniform(-0.025, 0.025)))
                t0 = time.perf_counter()
                destroyed, removed = self.destroy_operators[destroy_name](current_routes, adaptive_rate)
                perf['destroy'] += time.perf_counter() - t0
                t0 = time.perf_counter()
                candidate = None if destroyed is None else self.repair_operators[repair_name](destroyed, removed)
                perf['repair'] += time.perf_counter() - t0
                candidate_cost = None
                candidate_feasible = candidate is not None
                accepted = global_best_updated = False
                reward = 0.0
                if candidate is None:
                    no_improve += 1
                else:
                    t0 = time.perf_counter()
                    candidate_cost_before_ls = self.calculate_build_cost(candidate.keys()) + self.calculate_total_transport_cost(candidate)
                    perf['cost_and_accept'] += time.perf_counter() - t0
                    vnd_rounds = self._vnd_rounds_for_candidate(candidate_cost_before_ls, current_cost, iteration)
                    if vnd_rounds > 0:
                        t0 = time.perf_counter()
                        candidate = self._local_search_intra_routes(candidate, max_rounds=vnd_rounds)
                        perf['vnd'] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    candidate_cost = self.calculate_build_cost(candidate.keys()) + self.calculate_total_transport_cost(candidate)
                    delta = candidate_cost - current_cost
                    perf['cost_and_accept'] += time.perf_counter() - t0
                    accepted = delta <= 0 or random.random() < math.exp(-delta / max(temperature, 1e-12))
                    if _is_significant_best_improvement(candidate_cost, self.best_total_cost):
                        self.best_routes = deepcopy(candidate)
                        self.selected_depots = set(candidate.keys())
                        self.best_build_cost = self.calculate_build_cost(candidate.keys())
                        self.best_transport_cost = self.calculate_total_transport_cost(candidate)
                        self.best_total_cost = candidate_cost
                        self._push_elite(candidate, candidate_cost)
                        reward, no_improve, global_best_updated = 8.0, 0, True
                    elif accepted and candidate_cost < current_cost - 1e-10:
                        reward, no_improve = 4.0, max(0, no_improve - 1)
                    elif accepted:
                        reward, no_improve = 1.5, no_improve + 1
                    else:
                        no_improve += 1
                    if accepted:
                        current_routes, current_cost = candidate, candidate_cost
                self.destroy_scores[destroy_name] += reward
                self.repair_scores[repair_name] += reward
                temperature = max(T_end, temperature * cooling_rate)
                if iteration % self.segment_length == 0:
                    self._update_operator_weights()
                if no_improve >= self.stagnation_limit:
                    if self.elite_pool:
                        _, _, restart_routes = self.elite_pool[random.randrange(min(len(self.elite_pool), 4))]
                        current_routes = deepcopy(restart_routes)
                        current_cost = self.calculate_build_cost(current_routes.keys()) + self.calculate_total_transport_cost(current_routes)
                    else:
                        current_routes, current_cost = deepcopy(self.best_routes), self.best_total_cost
                    temperature, no_improve = max(temperature, T_start * 0.35), 0
                last_completed_iteration = iteration
                row = self._history_row(scenario=scenario_label, seed=random_seed, iteration=iteration, phase='iteration', current_cost=current_cost, candidate_cost=candidate_cost, temperature=temperature, destroy_rate=adaptive_rate, destroy_operator=destroy_name, repair_operator=repair_name, candidate_feasible=candidate_feasible, accepted=accepted, global_best_updated=global_best_updated, reward=reward, no_improve=no_improve, elapsed_seconds=elapsed_total())
                if history_writer is not None:
                    t0 = time.perf_counter()
                    history_writer.writerow(row)
                    if iteration % flush_interval == 0 or iteration == max_iter:
                        history_handle.flush()
                    perf['history_io'] += time.perf_counter() - t0
                if iteration % CLRP_ALNS_CHECKPOINT_INTERVAL == 0 or iteration == max_iter:
                    if history_handle is not None:
                        history_handle.flush()
                    save_checkpoint(False)
                if progress_callback is not None:
                    progress_callback(iteration, current_cost, self.best_total_cost)
                if print_interval and (iteration == 1 or iteration % print_interval == 0):
                    print(f'{prefix} iter {iteration}: Cur={current_cost:.2f} Best={self.best_total_cost:.2f} T={temperature:.3f} q={adaptive_rate:.2f} D={destroy_name} R={repair_name}')
            current_routes, current_cost = deepcopy(self.best_routes), self.best_total_cost
            feasible, violations = self.validate_solution(self.best_routes)
            if not feasible:
                raise RuntimeError('enhanced ALNS final solution failed constraint validation: ' + '; '.join(violations))
            save_checkpoint(True)
        except BaseException:
            save_checkpoint(False)
            raise
        finally:
            if history_handle is not None:
                history_handle.close()
        solve_time = elapsed_total()
        self.history_csv_path = history_csv_path
        if globals().get('CLRP_PROFILE_ALNS_TIME', False):
            measured = sum(perf.values())
            other = max(0.0, time.time() - run_start - measured)
            print(f"{prefix} time breakdown: destroy={perf['destroy']:.2f}s, repair={perf['repair']:.2f}s, VND={perf['vnd']:.2f}s, cost/accept={perf['cost_and_accept']:.2f}s, CSV={perf['history_io']:.2f}s, other={other:.2f}s")
        print(f'{prefix} === solve finished: Best={self.best_total_cost:.2f}, elapsed={solve_time:.2f}s ===')
        return (self.selected_depots, self.best_routes, self.best_build_cost, self.best_transport_cost, self.best_total_cost, solve_time)

def _setup_scientific_style():
    # Use Times New Roman for English text; enlarge labels on the three solution figures.
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.sans-serif': ['Times New Roman', 'Arial', 'DejaVu Sans'],
        'mathtext.fontset': 'stix',
        'axes.unicode_minus': False,
        'font.size': 14.0,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.7,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'lines.linewidth': 1.1,
        'lines.markersize': 3.5,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#b5b5b5',
        'legend.fancybox': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _apply_times_new_roman(ax):
    """Set axis titles, ticks, and existing text to Times New Roman."""
    for artist in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        if artist is not None:
            artist.set_fontname('Times New Roman')
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
    for text in ax.texts:
        text.set_fontname('Times New Roman')
    for child in ax.get_children():
        if hasattr(child, 'get_texts'):
            for t in child.get_texts():
                t.set_fontname('Times New Roman')

def _scientific_colormap(n):
    from colorsys import hsv_to_rgb
    from matplotlib.colors import to_hex
    n = max(0, int(n))
    if n == 0:
        return []
    colors = []
    for cmap_name in ('tab20', 'tab20b', 'tab20c'):
        cmap = plt.get_cmap(cmap_name)
        colors.extend((to_hex(cmap(i)) for i in range(cmap.N)))
    if n > len(colors):
        golden_ratio = 0.6180339887498949
        extra_count = n - len(colors)
        for i in range(extra_count):
            hue = (0.11 + i * golden_ratio) % 1.0
            saturation = (0.72, 0.88, 0.62)[i % 3]
            value = (0.78, 0.92)[i // 3 % 2]
            colors.append(to_hex(hsv_to_rgb(hue, saturation, value)))
    return colors[:n]

def _add_map_scale_bar(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    span = min(xmax - xmin, ymax - ymin)
    target = max(span * 0.16, 1.0)
    magnitude = 10 ** math.floor(math.log10(target))
    length = next((v * magnitude for v in (1, 2, 5, 10) if v * magnitude >= target))
    x0 = xmin + 0.06 * (xmax - xmin)
    y0 = ymin + 0.06 * (ymax - ymin)
    ax.plot([x0, x0 + length], [y0, y0], color='#202020', linewidth=2.0, solid_capstyle='butt', zorder=20)
    ax.plot([x0, x0], [y0 - 0.012 * span, y0 + 0.012 * span], color='#202020', linewidth=1.0, zorder=20)
    ax.plot([x0 + length, x0 + length], [y0 - 0.012 * span, y0 + 0.012 * span], color='#202020', linewidth=1.0, zorder=20)
    label = f'{length / 1000:g} km' if length >= 1000 else f'{length:g} m'
    ax.text(x0 + length / 2, y0 + 0.025 * span, label, ha='center', va='bottom',
            fontsize=16, color='#202020', zorder=20, fontname='Times New Roman',
            bbox=dict(facecolor='white', alpha=0.84, edgecolor='none', pad=0.7))

def _add_lonlat_envelope(ax, administrative_boundaries, boundary_shp=_ADMIN_BOUNDARY_SHP):
    if not administrative_boundaries:
        return
    points = np.vstack(administrative_boundaries)
    minx, miny = points.min(axis=0)
    maxx, maxy = points.max(axis=0)
    dx, dy = (maxx - minx, maxy - miny)
    # Leave a white margin for the frame and lon/lat labels, then draw the neatline inside;
    # the study-area outline then sits inside a regular rectangular frame instead of touching the edge.
    outer_pad_x, outer_pad_y = dx * 0.075, dy * 0.075
    frame_pad_x, frame_pad_y = dx * 0.030, dy * 0.030
    view_minx, view_miny = minx - outer_pad_x, miny - outer_pad_y
    view_maxx, view_maxy = maxx + outer_pad_x, maxy + outer_pad_y
    frame_minx, frame_miny = minx - frame_pad_x, miny - frame_pad_y
    frame_maxx, frame_maxy = maxx + frame_pad_x, maxy + frame_pad_y
    prj_path = os.path.splitext(boundary_shp)[0] + '.prj'
    source_crs = CRS.from_wkt(open(prj_path, encoding='utf-8-sig').read())
    transformer = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)
    corners = [(frame_minx, frame_miny), (frame_maxx, frame_miny), (frame_maxx, frame_maxy), (frame_minx, frame_maxy)]
    lonlat = [transformer.transform(x, y) for x, y in corners]
    from matplotlib.patches import Rectangle
    # Map neatline used in the manuscript: a solid frame with lon/lat labels at the four corners,
    # clearer than a dashed envelope and not confused with the study-area boundary.
    ax.set_xlim(view_minx, view_maxx)
    ax.set_ylim(view_miny, view_maxy)
    ax.add_patch(Rectangle((frame_minx, frame_miny), frame_maxx - frame_minx, frame_maxy - frame_miny, fill=False, edgecolor='#303030', linewidth=1.05, linestyle='-', zorder=10))
    alignments = [('right', 'top'), ('left', 'top'), ('left', 'bottom'), ('right', 'bottom')]
    for (x, y), (lon, lat), (ha, va) in zip(corners, lonlat, alignments):
        ax.annotate(f'{lon:.4f}°E\n{lat:.4f}°N', (x, y),
                    xytext=(8 if ha == 'left' else -8, 8 if va == 'bottom' else -8),
                    textcoords='offset points', ha=ha, va=va, fontsize=15, color='#303030',
                    zorder=11, fontname='Times New Roman',
                    bbox=dict(facecolor='white', alpha=0.92, edgecolor='none', pad=0.6))

def _add_prohibited_polygon_patches(ax, prohibited_polygons, facecolor='#d95f5f', edgecolor='#c54b4b', alpha=0.1, linewidth=0.65, linestyle='--', zorder=1):
    """Draw no-fly zones while keeping Polygon.interiors so holes left by railway-buffer unions are not filled as no-fly."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    for geometry in prohibited_polygons or []:
        for poly in _iter_polygonal_parts(geometry):
            if poly is None or poly.is_empty:
                continue
            rings = [np.asarray(poly.exterior.coords, dtype=float)]
            rings.extend((np.asarray(ring.coords, dtype=float) for ring in poly.interiors))
            ring_paths = [Path(ring) for ring in rings if len(ring) >= 4]
            if not ring_paths:
                continue
            path = Path.make_compound_path(*ring_paths) if len(ring_paths) > 1 else ring_paths[0]
            ax.add_patch(PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=linewidth, linestyle=linestyle, zorder=zorder, label='_nolegend_'))

def _solution_figure_paths(output_path):
    """Derive the map, legend, and route-length figure paths from a single output path."""
    output_path = os.path.abspath(os.fspath(output_path))
    directory = os.path.dirname(output_path)
    stem, ext = os.path.splitext(os.path.basename(output_path))
    ext = ext or '.png'
    if '_solution_' in stem:
        prefix, suffix = stem.split('_solution_', 1)
        make = lambda kind: os.path.join(directory, f'{prefix}_solution_{kind}_{suffix}{ext}')
        return {'map': make('map'), 'legend': make('legend'), 'route_length': make('route_length')}
    return {
        'map': os.path.join(directory, f'{stem}_map{ext}'),
        'legend': os.path.join(directory, f'{stem}_legend{ext}'),
        'route_length': os.path.join(directory, f'{stem}_route_length{ext}'),
    }

def _savefig_figure(fig, path, show_figure=False):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_path = f'{path}.tmp{os.path.splitext(path)[1] or ".png"}'
    fig.savefig(tmp_path, dpi=300, bbox_inches='tight', facecolor='white')
    os.replace(tmp_path, path)
    if show_figure:
        plt.show()
    plt.close(fig)
    return path

def show_result_CLRP(customers, coords, depot_build_cost, selected_depots, best_routes, build_cost, transport_cost, total_cost, prohibited_polygons, administrative_boundaries, output_path=None, show_figure=False, scenario_label='', random_seed=None, unplannable_details=None, navigation_data=None, navigation_distances=None):
    """Write three separate figures: spatial map, legend, and route-length summary."""
    _setup_scientific_style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    depot_list = sorted(selected_depots)
    n_depots = len(depot_list)
    route_colors = _scientific_colormap(n_depots)
    color_map = {depot: route_colors[i] for i, depot in enumerate(depot_list)}
    if output_path is None:
        output_path = os.path.join(_OUTPUT_DIR, f'{_output_stem("solution")}.png')
    paths = _solution_figure_paths(output_path)

    reason_styles = {
        'NO_DEPOT_IN_BLOCK': ('#2ca02c', 'No depot in block'),
        'OUT_OF_RANGE': ('#d62728', 'Beyond 6 km visibility range'),
        'TERRAIN_BLOCKED': ('#9467bd', 'Terrain-blocked waypoint'),
        'WAYPOINT_DEM_NODATA': ('#111111', 'Waypoint DEM NODATA'),
        'NO_VISIBLE_DEPOT': ('#ff7f0e', 'No visible depot'),
    }
    plotted_unreachable_labels = []
    grouped = {}
    for item in unplannable_details or []:
        grouped.setdefault(item.get('reason', 'NO_VISIBLE_DEPOT'), []).append((item['x'], item['y']))

    # ---------- 1) map (no legend) ----------
    fig_map = plt.figure(figsize=(12.6, 10.8), constrained_layout=False)
    ax1 = fig_map.add_subplot(111)
    if administrative_boundaries:
        for boundary in administrative_boundaries:
            ax1.plot(boundary[:, 0], boundary[:, 1], color='#3f3f3f', linewidth=0.9, alpha=0.95, zorder=0)
    ax1.scatter(coords[customers, 0], coords[customers, 1], color='#c9d0d6', s=4, marker='o', alpha=0.28, edgecolors='none', zorder=2)
    for reason, points in grouped.items():
        failed_xy = np.asarray(points, dtype=float)
        color, label = reason_styles.get(reason, ('#d62728', reason))
        ax1.scatter(failed_xy[:, 0], failed_xy[:, 1], marker='x', s=52, color=color, linewidths=1.45, zorder=6, label='_nolegend_')
        plotted_unreachable_labels.append((reason, color, label))
    if prohibited_polygons:
        _add_prohibited_polygon_patches(ax1, prohibited_polygons)
    for depot in depot_list:
        ax1.scatter(coords[depot, 0], coords[depot, 1], color=color_map[depot], s=108, marker='^', edgecolors='white', linewidths=1.15, zorder=7)
    for depot in best_routes:
        color = color_map[depot]
        for route in best_routes[depot]:
            route_coords = route_navigation_coordinates(navigation_data, route)
            if route_coords is None:
                route_coords = coords[route]
            ax1.plot(route_coords[:, 0], route_coords[:, 1], color=color, linewidth=1.05, alpha=0.82, zorder=3)
            via_nodes = route[1:-1]
            if via_nodes:
                via_xy = coords[via_nodes]
                ax1.scatter(via_xy[:, 0], via_xy[:, 1], s=20 * (2.0 / 3.0) ** 2, marker='o', facecolors='white', edgecolors=color, linewidths=0.65, zorder=5)
    for depot in depot_list:
        ax1.annotate(f'DC{depot}', (coords[depot, 0], coords[depot, 1]), xytext=(8, 8),
                     textcoords='offset points', fontsize=15, color='#202020', fontweight='bold',
                     fontname='Times New Roman', zorder=8,
                     bbox=dict(facecolor='white', alpha=0.88, edgecolor='none', pad=0.5))
    ax1.set_title('Spatial solution', loc='left', fontweight='bold', fontsize=20, pad=14,
                  fontname='Times New Roman')
    title_bits = [bit for bit in [scenario_label, None if random_seed is None else f'seed {random_seed}'] if bit]
    if title_bits:
        ax1.set_title(' · '.join(title_bits), loc='right', fontsize=14, color='#555555', pad=14,
                      fontname='Times New Roman')
    ax1.axis('equal')
    _add_map_scale_bar(ax1)
    _add_lonlat_envelope(ax1, administrative_boundaries)
    ax1.set_axis_off()
    _apply_times_new_roman(ax1)
    fig_map.subplots_adjust(bottom=0.04, top=0.94, left=0.02, right=0.98)
    _savefig_figure(fig_map, paths['map'], show_figure=show_figure)

    # ---------- 2) standalone legend ----------
    legend_handles = [
        Line2D([0], [0], color='#4c78a8', linewidth=1.2, marker='^', markersize=8,
               markerfacecolor='#4c78a8', markeredgecolor='white', markeredgewidth=0.6,
               label='Selected depot and route'),
    ]
    if prohibited_polygons:
        legend_handles.append(Patch(facecolor='#d95f5f', alpha=0.16, edgecolor='#c54b4b',
                                    linewidth=0.65, linestyle='--', label='No-fly zone'))
    if administrative_boundaries:
        legend_handles.append(Line2D([0], [0], color='#3f3f3f', linewidth=0.9, label='Research-area boundary'))
    legend_handles.append(Line2D([0], [0], color='#555555', marker='o', markersize=5.5,
                                 markerfacecolor='white', markeredgecolor='#555555', linewidth=0,
                                 label='Route waypoint'))
    for _, color, label in plotted_unreachable_labels:
        legend_handles.append(Line2D([0], [0], color=color, marker='x', markersize=8, linewidth=0,
                                     markeredgewidth=1.4, label=label))
    legend_height = max(3.2, 0.85 + 0.62 * len(legend_handles))
    fig_leg = plt.figure(figsize=(6.8, legend_height), constrained_layout=False)
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis('off')
    ax_leg.set_title('Legend', loc='left', fontweight='bold', fontsize=20, pad=10,
                     fontname='Times New Roman')
    legend = ax_leg.legend(
        handles=legend_handles, loc='upper left', bbox_to_anchor=(0.0, 0.92),
        ncol=1, frameon=True, facecolor='white', framealpha=1.0,
        edgecolor='#a8a8a8', borderpad=1.0, labelspacing=1.0,
        handlelength=2.6, handletextpad=0.9, fontsize=17,
        prop={'family': 'Times New Roman', 'size': 17},
    )
    for text in legend.get_texts():
        text.set_fontname('Times New Roman')
        text.set_fontsize(17)
    _apply_times_new_roman(ax_leg)
    fig_leg.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.06)
    _savefig_figure(fig_leg, paths['legend'], show_figure=show_figure)

    # ---------- 3) route-length summary ----------
    depot_total_km = []
    depot_route_counts = []
    all_route_km = []
    def _route_length_m(route):
        if navigation_distances is not None:
            return sum((float(navigation_distances[route[i]][route[i + 1]]) for i in range(len(route) - 1)))
        pts = coords[route]
        return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))

    for depot in depot_list:
        lengths_m = []
        for route in best_routes.get(depot, []):
            length_m = _route_length_m(route)
            lengths_m.append(length_m)
            all_route_km.append(length_m / 1000.0)
        depot_total_km.append(sum(lengths_m) / 1000.0)
        depot_route_counts.append(len(lengths_m))
    order = np.argsort(depot_total_km)
    ordered_depots = [depot_list[i] for i in order]
    ordered_totals = [depot_total_km[i] for i in order]
    ordered_counts = [depot_route_counts[i] for i in order]
    ordered_colors = [color_map[d] for d in ordered_depots]
    chart_h = max(6.8, 0.30 * max(n_depots, 1) + 2.4)
    fig_len = plt.figure(figsize=(8.6 * 2.0 / 3.0, chart_h), constrained_layout=False)
    ax2 = fig_len.add_subplot(111)
    y_pos = np.arange(n_depots)
    ax2.barh(y_pos, ordered_totals, height=0.62, color=ordered_colors, alpha=0.88, edgecolor='white', linewidth=0.4)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'DC{d}' for d in ordered_depots], fontsize=15, fontname='Times New Roman')
    ax2.set_xlabel('Total route length (km)', fontsize=18, fontname='Times New Roman')
    ax2.set_title('Route length summary', loc='left', fontweight='bold', fontsize=20, pad=14,
                  fontname='Times New Roman')
    if all_route_km:
        ax2.set_title(f'n={len(all_route_km)} routes', loc='right', fontsize=15, color='#555555', pad=14,
                      fontname='Times New Roman')
    ax2.tick_params(axis='x', labelsize=15)
    ax2.grid(axis='x', color='#e0e0e0', linewidth=0.45, alpha=0.8)
    ax2.grid(axis='y', visible=False)
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color('#303030')
        spine.set_linewidth(0.9)
    max_total = max(ordered_totals) if ordered_totals else 1.0
    # Leave enough axis space for end-of-bar labels so they do not overflow the right spine.
    ax2.set_xlim(0, max_total * 1.72)
    for y, total, n_routes in zip(y_pos, ordered_totals, ordered_counts):
        ax2.text(total + max_total * 0.025, y, f'{total:.1f} km  |  {n_routes} routes',
                 ha='left', va='center', fontsize=14, color='#424242', fontname='Times New Roman')
    _apply_times_new_roman(ax2)
    fig_len.subplots_adjust(bottom=0.09, top=0.91, left=0.16, right=0.96)
    _savefig_figure(fig_len, paths['route_length'], show_figure=show_figure)
    return paths

def _scenario_label_from_polygons(prohibited_polygons):
    return 'with_no_fly_zone' if prohibited_polygons else 'without_no_fly_zone'

def _sanitize_filename(value):
    value = re.sub('[^A-Za-z0-9_.-]+', '_', str(value)).strip('_.')
    return value or 'experiment'

class _BlockProgressBar:

    def __init__(self, seed, block_id, total):
        self.seed, self.block_id, self.total = (int(seed), int(block_id), max(1, int(total)))
        self.pid, self.last, self.finished = (os.getpid(), -1, False)

    def update(self, iteration, current_cost, best_cost):
        percent = int(100 * int(iteration) / self.total)
        if percent == self.last and iteration != self.total:
            return
        self.last = percent
        filled = int(24 * percent / 100)
        bar = '#' * filled + '-' * (24 - filled)
        print(f'\r[progress|pid={self.pid}|seed={self.seed}|block={self.block_id + 1}] [{bar}] {percent:3d}% ({iteration}/{self.total}) Cur={current_cost:.2f} Best={best_cost:.2f}', end='', flush=True)
        if iteration >= self.total:
            self.finished = True
            print(flush=True)

    def close(self):
        if self.last >= 0 and (not self.finished):
            print(flush=True)

def build_free_flight_blocks(coords, depot_candidates, customers, administrative_boundary_shp, route_prohibited_polygons):
    boundary = load_polygonal_area_geometry(administrative_boundary_shp)
    forbidden = unary_union([p for p in route_prohibited_polygons or [] if p is not None and (not p.is_empty)])
    safe_forbidden = forbidden.buffer(CLRP_NAVIGATION_CLEARANCE) if not forbidden.is_empty else forbidden
    free = boundary.difference(safe_forbidden) if not safe_forbidden.is_empty else boundary
    blocks = [p for p in _iter_polygonal_parts(free) if p.area > 1e-06]
    if not blocks:
        raise ValueError('no free-flight area remains after applying the no-fly safety buffer.')
    node_block = np.full(len(coords), -1, dtype=int)
    for node in list(depot_candidates) + list(customers):
        point = Point(coords[node])
        for block_id, block in enumerate(blocks):
            if block.covers(point):
                node_block[node] = block_id
                break
    block_info = []
    for block_id, block in enumerate(blocks):
        ds = [d for d in depot_candidates if node_block[d] == block_id]
        cs = [c for c in customers if node_block[c] == block_id]
        block_info.append({'block_id': block_id, 'geometry': block, 'depots': ds, 'customers': cs})
    bad = [item for item in block_info if item['customers'] and (not item['depots'])]
    if bad:
        print('[no-fly partition] ' + '; '.join((f"block {item['block_id'] + 1} has {len(item['customers'])} waypoint(s) but no candidate depot" for item in bad)))
    print('[no-fly partition] ' + '; '.join((f"block {item['block_id'] + 1}: depots={len(item['depots'])}, waypoints={len(item['customers'])}" for item in block_info if item['customers'])))
    return (node_block, block_info)

def _load_problem_once(no_fly_shp=_DEFAULT_NO_FLY_SHP, administrative_boundary_shp=None):
    administrative_boundary_shp = administrative_boundary_shp or _ADMIN_BOUNDARY_SHP
    route_prohibited_polygons = load_prohibited_flight_areas(no_fly_shp)
    same_exclusion_source = bool(no_fly_shp) and os.path.normcase(os.path.abspath(no_fly_shp)) == os.path.normcase(os.path.abspath(_WAYPOINT_EXCLUSION_SHP))
    if route_prohibited_polygons:
        waypoint_exclusion_polygons = list(_iter_polygonal_parts(unary_union(route_prohibited_polygons).buffer(CLRP_NAVIGATION_CLEARANCE)))
    elif same_exclusion_source:
        waypoint_exclusion_polygons = list(route_prohibited_polygons)
    else:
        waypoint_exclusion_polygons = load_prohibited_flight_areas(_WAYPOINT_EXCLUSION_SHP)

    data = load_zunhua_UAV_data_for_CLRP(
        administrative_boundary_shp=administrative_boundary_shp,
        prohibited_polygons=waypoint_exclusion_polygons,
        route_prohibited_polygons=route_prohibited_polygons,
        n_selected_depots=CLRP_N_SELECTED_DEPOTS,
        max_vehicles_per_depot=50,
        vehicle_cap=100,
        max_route_length=CLRP_MAX_ROUTE_LENGTH,
    )
    (n_depot_candidates, n_selected_depots, n_customers, max_vehicles_per_depot,
     vehicle_cap, depot_candidates, customers, all_nodes, coords, depot_build_cost,
     demands, base_dist_matrix) = data

    # Build free-flight connected blocks first; node pairs in different blocks never share a route, so detours are skipped.
    administrative_boundaries = load_administrative_boundaries(administrative_boundary_shp)
    node_block, block_info = build_free_flight_blocks(
        coords, depot_candidates, customers, administrative_boundary_shp,
        route_prohibited_polygons,
    )

    navigation_config = {
        'navigation_clearance': float(CLRP_NAVIGATION_CLEARANCE),
        'boundary_step': float(CLRP_NAVIGATION_BOUNDARY_STEP),
        'boundary_neighbors': int(CLRP_NAVIGATION_BOUNDARY_NEIGHBORS),
        'boundary_link_neighbors': int(CLRP_NAVIGATION_BOUNDARY_LINK_NEIGHBORS),
        'navigation_margin': float(CLRP_NAVIGATION_MARGIN),
        'max_boundary_nodes': int(CLRP_NAVIGATION_MAX_BOUNDARY_NODES),
        'terminal_candidate_limit': int(CLRP_NAVIGATION_TERMINAL_CANDIDATE_LIMIT),
        'target_chunk_size': int(CLRP_NAVIGATION_TARGET_CHUNK_SIZE),
        'max_edge_length': float(CLRP_MAX_ROUTE_LENGTH),
        'scenario_has_no_fly_zone': bool(route_prohibited_polygons),
    }
    navigation_key = _matrix_checkpoint_key(
        'navigation_scalable',
        arrays=(coords, node_block),
        config=navigation_config,
        source_paths=(administrative_boundary_shp, no_fly_shp),
    )
    cached_navigation = _load_navigation_checkpoint(navigation_key, len(all_nodes))
    if cached_navigation is None:
        horizontal_navigation_distances, navigation_data = build_navigation_distances(
            coords,
            route_prohibited_polygons,
            clearance=CLRP_NAVIGATION_CLEARANCE,
            boundary_step=CLRP_NAVIGATION_BOUNDARY_STEP,
            boundary_neighbors=CLRP_NAVIGATION_BOUNDARY_NEIGHBORS,
            boundary_link_neighbors=CLRP_NAVIGATION_BOUNDARY_LINK_NEIGHBORS,
            navigation_margin=CLRP_NAVIGATION_MARGIN,
            component_ids=node_block,
            max_edge_length=CLRP_MAX_ROUTE_LENGTH,
            max_boundary_nodes=CLRP_NAVIGATION_MAX_BOUNDARY_NODES,
            terminal_candidate_limit=CLRP_NAVIGATION_TERMINAL_CANDIDATE_LIMIT,
            target_chunk_size=CLRP_NAVIGATION_TARGET_CHUNK_SIZE,
        )
        horizontal_navigation_distances = np.asarray(horizontal_navigation_distances, dtype=np.float32)
        no_fly_crossing = np.not_equal.outer(node_block, node_block)
        _save_navigation_checkpoint(
            navigation_key,
            horizontal_navigation_distances,
            no_fly_crossing,
            navigation_data,
            metadata={
                'n_nodes': int(len(all_nodes)),
                'description': '2D near-tangent no-fly detour distances built per free-flight block',
                'navigation_config': navigation_config,
            },
        )
    else:
        horizontal_navigation_distances, no_fly_crossing, navigation_data = cached_navigation
        no_fly_crossing |= np.not_equal.outer(node_block, node_block)

    ground_elevations = np.asarray(_LAST_NODE_GROUND_ELEVATIONS, dtype=np.float32)
    flight_heights = np.asarray(_LAST_NODE_FLIGHT_HEIGHTS, dtype=np.float32)
    visibility_matrix = np.asarray(_LAST_DEPOT_WAYPOINT_VISIBILITY, dtype=bool)
    n_all = len(all_nodes)
    scenario = _scenario_label_from_polygons(route_prohibited_polygons)

    flight_key = _matrix_checkpoint_key(
        'flight_AB_BA_BB_scalable',
        arrays=(coords, flight_heights, horizontal_navigation_distances, node_block),
        config={
            'dem_clearance': float(CLRP_DEM_CLEARANCE),
            'profile_step': float(CLRP_DEM_PROFILE_STEP),
            'navigation_clearance': float(CLRP_NAVIGATION_CLEARANCE),
            'scenario': scenario,
            'matrix_dtype': 'float32',
        },
        source_paths=(administrative_boundary_shp, _DEM_TIF, no_fly_shp),
    )
    cached_flight = _load_matrix_checkpoint(
        'flight_AB_BA_BB_scalable',
        flight_key,
        expected_shapes={name: (n_all, n_all) for name in ('total', 'ascent', 'horizontal', 'descent', 'cruise')},
    )
    if cached_flight is not None:
        cached_arrays, _ = cached_flight
        cached_total = cached_arrays['total'].astype(np.float32, copy=False)
        ascent_3d = cached_arrays['ascent'].astype(np.float32, copy=False)
        horizontal_3d = cached_arrays['horizontal'].astype(np.float32, copy=False)
        descent_3d = cached_arrays['descent'].astype(np.float32, copy=False)
        cruise_heights = cached_arrays['cruise'].astype(np.float32, copy=False)

        # Older checkpoints may have stored total from float64-sum-then-cast. Do not reuse it;
        # rebuild total from the three components as the only source of truth.
        dist_matrix = _compose_total_flight_distance(
            ascent_3d, horizontal_3d, descent_3d
        )
        cache_report = _distance_matrix_consistency_report(cached_total, dist_matrix)
        if not cache_report['consistent']:
            print(
                '[flight-matrix checkpoint] rebuilt total from ascent+horizontal+descent; '
                f"old-total max difference={cache_report['max_difference']:.9f} m, "
                f"out-of-tolerance entries={cache_report['mismatch_count']}, "
                f"finite-pattern mismatch={cache_report['finite_pattern_mismatch']}."
            )
    else:
        dist_matrix = np.full((n_all, n_all), np.inf, dtype=np.float32)
        ascent_3d = np.full((n_all, n_all), np.inf, dtype=np.float32)
        horizontal_3d = np.full((n_all, n_all), np.inf, dtype=np.float32)
        descent_3d = np.full((n_all, n_all), np.inf, dtype=np.float32)
        cruise_heights = np.full((n_all, n_all), np.nan, dtype=np.float32)
        for matrix in (dist_matrix, ascent_3d, horizontal_3d, descent_3d):
            np.fill_diagonal(matrix, 0.0)
        np.fill_diagonal(cruise_heights, flight_heights)
        dem_model = DEMTerrainModel(_DEM_TIF, _read_prj_crs(administrative_boundary_shp))
        active_blocks = [block for block in block_info if block['depots'] or block['customers']]
        for order, block in enumerate(active_blocks, 1):
            block_nodes = list(block['depots']) + list(block['customers'])
            index = np.asarray(block_nodes, dtype=int)
            print(f"[DEM 3D matrix] block {block['block_id'] + 1} ({order}/{len(active_blocks)}): nodes={len(index)}")
            matrices = build_three_dimensional_flight_matrices(
                np.asarray(coords)[index],
                flight_heights[index],
                dem_model,
                navigation_data=navigation_data,
                horizontal_distances=horizontal_navigation_distances[np.ix_(index, index)],
                node_ids=index,
                global_coords=coords,
            )
            ix = np.ix_(index, index)
            for target, local in zip(
                (dist_matrix, ascent_3d, horizontal_3d, descent_3d, cruise_heights),
                matrices,
            ):
                target[ix] = np.asarray(local, dtype=np.float32)
        # Newly computed checkpoints also rebuild total from the three float32 components so saved files stay consistent.
        dist_matrix = _compose_total_flight_distance(
            ascent_3d, horizontal_3d, descent_3d
        )
        _save_matrix_checkpoint(
            'flight_AB_BA_BB_scalable',
            flight_key,
            arrays={
                'total': dist_matrix,
                'ascent': ascent_3d,
                'horizontal': horizontal_3d,
                'descent': descent_3d,
                'cruise': cruise_heights,
            },
            metadata={
                'n_nodes': int(n_all),
                'n_depots': int(n_depot_candidates),
                'n_waypoints': int(n_customers),
                'description': 'block-wise float32 directed 3D flight matrices',
            },
        )

    return {
        'n_depot_candidates': n_depot_candidates,
        'n_selected_depots': n_selected_depots,
        'n_customers': n_customers,
        'max_vehicles_per_depot': max_vehicles_per_depot,
        'vehicle_cap': vehicle_cap,
        'depot_candidates': depot_candidates,
        'customers': customers,
        'all_nodes': all_nodes,
        'coords': coords,
        'depot_build_cost': depot_build_cost,
        'demands': demands,
        'dist_matrix': dist_matrix,
        'navigation_distances': dist_matrix,
        'navigation_data': navigation_data,
        'unplannable_waypoints': list(_LAST_UNPLANNABLE_WAYPOINTS),
        'partition_unplannable_waypoints': list(_LAST_PARTITION_UNPLANNABLE_WAYPOINTS),
        'unplannable_details': list(_LAST_UNPLANNABLE_DETAILS),
        'ground_elevations': ground_elevations,
        'flight_heights': flight_heights,
        'visibility_matrix': visibility_matrix,
        'ascent_distances': ascent_3d,
        'horizontal_distances': horizontal_3d,
        'descent_distances': descent_3d,
        'vertical_distances': ascent_3d + descent_3d,
        'cruise_heights': cruise_heights,
        'prohibited_polygons': waypoint_exclusion_polygons,
        'no_fly_crossing': no_fly_crossing,
        'administrative_boundaries': administrative_boundaries,
        'scenario_label': scenario,
        'node_block': node_block,
        'free_flight_blocks': block_info,
        'no_fly_shp': os.path.abspath(no_fly_shp) if no_fly_shp else '',
        'administrative_boundary_shp': os.path.abspath(administrative_boundary_shp),
    }

def _build_solver(problem):
    return ExperimentALNS_CLRP(depot_candidates=problem['depot_candidates'], customers=problem['customers'], all_nodes=problem['all_nodes'], coords=problem['coords'], depot_build_cost=problem['depot_build_cost'], demands=problem['demands'], vehicle_cap=problem['vehicle_cap'], dist_matrix=problem['dist_matrix'], n_selected_depots=problem['n_selected_depots'], max_vehicles_per_depot=problem['max_vehicles_per_depot'], max_route_length=CLRP_MAX_ROUTE_LENGTH, no_fly_crossing=problem['no_fly_crossing'], navigation_distances=problem['navigation_distances'], visibility_matrix=problem['visibility_matrix'], ground_elevations=problem['ground_elevations'], flight_heights=problem['flight_heights'], ascent_distances=problem['ascent_distances'], horizontal_distances=problem['horizontal_distances'], descent_distances=problem['descent_distances'], vertical_distances=problem['vertical_distances'], cruise_heights=problem['cruise_heights'], segment_length=CLRP_SEGMENT_LENGTH, local_search_interval=CLRP_LOCAL_SEARCH_INTERVAL, elite_size=CLRP_ELITE_SIZE, stagnation_limit=CLRP_STAGNATION_LIMIT, min_destroy_rate=CLRP_MIN_DESTROY_RATE, max_destroy_rate=CLRP_MAX_DESTROY_RATE, max_destroy_customers=CLRP_MAX_DESTROY_CUSTOMERS)

def _make_local_block_problem(problem, block):
    depots = list(block['depots'])
    customers = list(block['customers'])
    if not customers:
        return None
    if not depots:
        raise ValueError(f"free-flight block {block['block_id'] + 1} has waypoints but no candidate depot")
    local_to_global = depots + customers
    index = np.asarray(local_to_global, dtype=int)
    matrix_index = np.ix_(index, index)
    local_build_cost = {local_id: float(problem['depot_build_cost'][global_id]) for local_id, global_id in enumerate(depots)}
    return {'block_id': int(block['block_id']), 'local_to_global': local_to_global, 'depot_candidates': list(range(len(depots))), 'customers': list(range(len(depots), len(local_to_global))), 'all_nodes': list(range(len(local_to_global))), 'coords': np.asarray(problem['coords'])[index].copy(), 'depot_build_cost': local_build_cost, 'demands': np.asarray(problem['demands'])[index].copy(), 'dist_matrix': np.asarray(problem['dist_matrix'])[matrix_index].copy(), 'navigation_distances': np.asarray(problem['navigation_distances'])[matrix_index].copy(), 'visibility_matrix': np.asarray(problem['visibility_matrix'])[matrix_index].copy(), 'ground_elevations': np.asarray(problem['ground_elevations'])[index].copy(), 'flight_heights': np.asarray(problem['flight_heights'])[index].copy(), 'ascent_distances': np.asarray(problem['ascent_distances'])[matrix_index].copy(), 'horizontal_distances': np.asarray(problem['horizontal_distances'])[matrix_index].copy(), 'descent_distances': np.asarray(problem['descent_distances'])[matrix_index].copy(), 'vertical_distances': np.asarray(problem['vertical_distances'])[matrix_index].copy(), 'cruise_heights': np.asarray(problem['cruise_heights'])[matrix_index].copy(), 'no_fly_crossing': np.asarray(problem['no_fly_crossing'])[matrix_index].copy(), 'n_selected_depots': None, 'max_vehicles_per_depot': problem['max_vehicles_per_depot'], 'vehicle_cap': problem['vehicle_cap'], 'scenario_label': problem['scenario_label'], 'no_fly_shp': problem['no_fly_shp'], 'administrative_boundary_shp': problem['administrative_boundary_shp']}

def _write_json_atomic(path, payload):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path

def _routes_to_json(routes):
    return {str(int(depot)): [[int(node) for node in route] for route in depot_routes] for depot, depot_routes in routes.items()}

def _routes_from_json(routes):
    return {int(depot): [[int(node) for node in route] for route in depot_routes] for depot, depot_routes in routes.items()}

def save_solution_result(path, selected_depots, routes, build_cost, transport_cost, total_cost, scenario, seed, no_fly_shp='', administrative_boundary_shp='', solve_time=None, iteration_csvs=None, block_result_files=None, result_kind='seed_solution', extra_metadata=None):
    payload = {'format_version': 1, 'result_kind': result_kind, 'created_at': datetime.now().isoformat(timespec='seconds'), 'scenario': str(scenario), 'seed': int(seed), 'selected_depots': sorted((int(x) for x in selected_depots)), 'routes': _routes_to_json(routes), 'build_cost': float(build_cost), 'transport_cost': float(transport_cost), 'total_cost': float(total_cost), 'solve_time_seconds': None if solve_time is None else float(solve_time), 'iteration_csvs': [os.path.abspath(p) for p in iteration_csvs or []], 'block_result_files': [os.path.abspath(p) for p in block_result_files or []], 'no_fly_shp': os.path.abspath(no_fly_shp) if no_fly_shp else '', 'administrative_boundary_shp': os.path.abspath(administrative_boundary_shp) if administrative_boundary_shp else os.path.abspath(_ADMIN_BOUNDARY_SHP)}
    if extra_metadata:
        payload.update(extra_metadata)
    return _write_json_atomic(path, payload)

def plot_saved_solution(result_json, output_path=None, show_figure=False):
    result_json = os.path.abspath(os.fspath(result_json))
    with open(result_json, 'r', encoding='utf-8') as file_obj:
        result = json.load(file_obj)
    problem = _load_problem_once(no_fly_shp=result.get('no_fly_shp', _DEFAULT_NO_FLY_SHP), administrative_boundary_shp=result.get('administrative_boundary_shp', _ADMIN_BOUNDARY_SHP))
    selected = [int(x) for x in result['selected_depots']]
    routes = _routes_from_json(result['routes'])
    if output_path is None:
        stem = os.path.splitext(os.path.basename(result_json))[0].replace('_solution_result_', '_solution_')
        output_path = os.path.join(os.path.dirname(result_json), f'{stem}.png')
    image_paths = show_result_CLRP(problem['customers'], problem['coords'], problem['depot_build_cost'], selected, routes, float(result['build_cost']), float(result['transport_cost']), float(result['total_cost']), problem['prohibited_polygons'], problem['administrative_boundaries'], output_path=output_path, show_figure=show_figure, scenario_label=result.get('scenario', ''), random_seed=result.get('seed'), unplannable_details=problem['unplannable_details'], navigation_data=problem['navigation_data'], navigation_distances=problem['navigation_distances'])
    print(f'redrawn from saved result: map={image_paths["map"]}')
    print(f'redrawn from saved result: legend={image_paths["legend"]}')
    print(f'redrawn from saved result: route length={image_paths["route_length"]}')
    return image_paths

def _run_single_seed_worker(payload):
    problem = payload['problem']
    seed = int(payload['seed'])
    max_iter = int(payload['max_iter'])
    output_dir = os.path.abspath(payload['output_dir'])
    scenario = problem['scenario_label']
    scenario_file = _sanitize_filename(scenario)
    os.makedirs(output_dir, exist_ok=True)
    _apply_clrp_random_seed(seed * 1000003 + int(problem['block_id']))
    progress_bar = _BlockProgressBar(seed, problem['block_id'], max_iter)
    solver = _build_solver(problem)
    block_number = int(problem['block_id']) + 1
    iteration_csv = os.path.join(output_dir, f'{_output_stem("iterations", scenario_file, f"seed_{seed}", f"block_{block_number}")}.csv')
    checkpoint_path = os.path.join(output_dir, f'{_output_stem("alns_resume", scenario_file, f"seed_{seed}", f"block_{block_number}")}.pkl')
    selected_depots, best_routes, build_cost, transport_cost, total_cost, solve_time = solver.alns_optimize(max_iter=max_iter, destroy_rate=CLRP_DESTROY_RATE, history_csv_path=iteration_csv, checkpoint_path=checkpoint_path, random_seed=seed, scenario_label=scenario, print_interval=0, progress_callback=progress_bar.update)
    progress_bar.close()
    local_to_global = problem['local_to_global']
    global_selected = [local_to_global[d] for d in selected_depots]
    global_routes = {local_to_global[depot]: [[local_to_global[node] for node in route] for route in routes] for depot, routes in best_routes.items()}
    route_lengths = [solver.calculate_route_length_actual(route) for routes in best_routes.values() for route in routes]
    route_loads = [solver._route_load(route) for routes in best_routes.values() for route in routes]
    block_result_json = os.path.join(output_dir, f'{_output_stem("block_result", scenario_file, f"seed_{seed}", f"block_{block_number}")}.json')
    refinement_history = []
    for record in getattr(solver, 'initial_facility_refinement_history', []):
        item = dict(record)
        item['old_group'] = [int(local_to_global[node]) for node in record.get('old_group', [])]
        item['replacement'] = [int(local_to_global[node]) for node in record.get('replacement', [])]
        refinement_history.append(item)
    save_solution_result(block_result_json, global_selected, global_routes, build_cost, transport_cost, total_cost, scenario, seed, no_fly_shp=problem['no_fly_shp'], administrative_boundary_shp=problem['administrative_boundary_shp'], solve_time=solve_time, iteration_csvs=[iteration_csv], result_kind='block_solution', extra_metadata={'initial_facility_refinement_history': refinement_history})
    return {'scenario': scenario, 'seed': seed, 'block_id': int(problem['block_id']), 'status': 'success', 'initial_total_cost': f'{solver.initial_total_cost:.10f}', 'best_total_cost': f'{total_cost:.10f}', 'build_cost': f'{build_cost:.10f}', 'transport_cost': f'{transport_cost:.10f}', 'improvement_absolute': f'{solver.initial_total_cost - total_cost:.10f}', 'improvement_percent': f'{(solver.initial_total_cost - total_cost) / solver.initial_total_cost * 100.0:.8f}', 'solve_time_seconds': f'{solve_time:.6f}', 'n_open_depots': len(selected_depots), 'selected_depots': ';'.join(map(str, sorted(global_selected))), 'n_routes': len(route_lengths), 'mean_route_load': f'{np.mean(route_loads):.8f}', 'max_route_length_km': f'{max(route_lengths) / 1000.0:.8f}', 'iteration_csv': iteration_csv, 'block_result_json': block_result_json, 'global_selected_depots': global_selected, 'global_routes': global_routes}

def _write_dict_rows(path, rows):
    if not rows:
        return path
    all_fields = []
    for row in rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)
    return path

def _merge_iteration_csvs(summary_rows, output_path):
    csv_paths = [row.get('iteration_csv') for row in summary_rows if row.get('status') == 'success' and row.get('iteration_csv')]
    if not csv_paths:
        return None
    writer = None
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as out_f:
        for path in csv_paths:
            with open(path, 'r', newline='', encoding='utf-8-sig') as in_f:
                reader = csv.DictReader(in_f)
                if writer is None:
                    writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
    return output_path

def export_candidate_pool_csv(problem, output_dir):
    scenario = _sanitize_filename(problem['scenario_label'])
    output_path = os.path.join(output_dir, f'{_output_stem("candidate_takeoff_landing_points", scenario)}.csv')
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'x', 'y', 'build_cost', 'candidate_source'])
        for depot in problem['depot_candidates']:
            writer.writerow([depot, f"{problem['coords'][depot][0]:.6f}", f"{problem['coords'][depot][1]:.6f}", f"{problem['depot_build_cost'][depot]:.6f}", 'FCLRP_Zunhua_data/zunhua_random_depots_2000'])
    return output_path

def export_visibility_matrix_csv(problem, output_dir):
    visibility = problem.get('visibility_matrix')
    if visibility is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{_output_stem("depot_waypoint_visibility_matrix")}.csv')
    depots = list(problem['depot_candidates'])
    customers = list(problem['customers'])
    with open(path, 'w', newline='', encoding='utf-8-sig') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(['depot_node_id'] + [f'waypoint_{c}' for c in customers])
        for depot in depots:
            writer.writerow([depot] + [int(bool(visibility[depot][c])) for c in customers])
    return path

def export_flight_matrix_checkpoint(problem, output_dir):
    ascent = problem.get('ascent_distances')
    horizontal = problem.get('horizontal_distances')
    descent = problem.get('descent_distances')
    visibility = problem.get('visibility_matrix')
    if ascent is None or horizontal is None or descent is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    depots = np.asarray(problem['depot_candidates'], dtype=np.int64)
    waypoints = np.asarray(problem['customers'], dtype=np.int64)
    ab = np.ix_(depots, waypoints)
    ba = np.ix_(waypoints, depots)
    bb = np.ix_(waypoints, waypoints)
    output_path = os.path.join(output_dir, f'{_output_stem("flight_matrix_checkpoint_AB_BA_BB")}.npz')
    with open(output_path + '.tmp', 'wb') as file_obj:
        np.savez_compressed(file_obj, depot_node_ids=depots, waypoint_node_ids=waypoints, visibility_AB=np.asarray(visibility)[ab].astype(np.uint8) if visibility is not None else np.ones((len(depots), len(waypoints)), dtype=np.uint8), ascent_AB=np.asarray(ascent)[ab], horizontal_AB=np.asarray(horizontal)[ab], descent_AB=np.asarray(descent)[ab], ascent_BA=np.asarray(ascent)[ba], horizontal_BA=np.asarray(horizontal)[ba], descent_BA=np.asarray(descent)[ba], ascent_BB=np.asarray(ascent)[bb], horizontal_BB=np.asarray(horizontal)[bb], descent_BB=np.asarray(descent)[bb], ascent_full=np.asarray(ascent), horizontal_full=np.asarray(horizontal), descent_full=np.asarray(descent), total_full=np.asarray(problem['navigation_distances']), cruise_height_full=np.asarray(problem['cruise_heights']))
    os.replace(output_path + '.tmp', output_path)
    manifest_path = os.path.join(output_dir, f'{_output_stem("flight_matrix_checkpoint_manifest")}.json')
    with open(manifest_path, 'w', encoding='utf-8') as file_obj:
        json.dump({'checkpoint': output_path, 'direction_note': 'Ascent and descent are directed; return legs B→A cannot reuse A→B, so BA is stored separately. The horizontal matrix is symmetric in the normal case.', 'matrix_rule': 'total = ascent + horizontal + descent', 'depot_height_rule': f'DEM+{CLRP_DEPOT_HEIGHT_AGL:g}m', 'waypoint_height_rule': f'DEM+{CLRP_WAYPOINT_HEIGHT_AGL:g}m', 'terrain_clearance_rule': f'highest DEM along the profile +{CLRP_DEM_CLEARANCE:g}m', 'n_depots': int(len(depots)), 'n_waypoints': int(len(waypoints))}, file_obj, ensure_ascii=False, indent=2)
    return output_path

def export_node_elevations_csv(problem, output_dir):
    ground = problem.get('ground_elevations')
    heights = problem.get('flight_heights')
    if ground is None or heights is None:
        return None
    path = os.path.join(output_dir, f'{_output_stem("node_dem_and_flight_heights")}.csv')
    depots = set(problem['depot_candidates'])
    with open(path, 'w', newline='', encoding='utf-8-sig') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(['node_id', 'node_type', 'x', 'y', 'dem_ground_m', 'flight_height_m', 'height_agl_m'])
        for node in problem['all_nodes']:
            writer.writerow([node, 'depot' if node in depots else 'waypoint', float(problem['coords'][node][0]), float(problem['coords'][node][1]), float(ground[node]), float(heights[node]), float(heights[node] - ground[node])])
    return path

def export_unplannable_waypoints_csv(problem, output_dir):
    details = problem.get('unplannable_details') or []
    if not details:
        return None
    path = os.path.join(output_dir, f'{_output_stem("unplannable_waypoints")}.csv')
    fields = ['original_waypoint_id', 'x', 'y', 'reason', 'block_id']
    with open(path, 'w', newline='', encoding='utf-8-sig') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for item in details:
            writer.writerow({key: item.get(key) for key in fields})
    return path

def export_route_3d_segments_csv(problem, routes, output_path):
    ascent_matrix = problem.get('ascent_distances')
    horizontal = problem.get('horizontal_distances')
    descent_matrix = problem.get('descent_distances')
    vertical = problem.get('vertical_distances')
    cruise = problem.get('cruise_heights')
    heights = problem.get('flight_heights')
    total = problem.get('navigation_distances')
    if ascent_matrix is None or horizontal is None or descent_matrix is None or (cruise is None) or (heights is None):
        return None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(['depot', 'route_index', 'segment_index', 'from_node', 'to_node', 'from_height_m', 'to_height_m', 'cruise_height_m', 'ascent_m', 'horizontal_m', 'descent_m', 'vertical_total_m', 'flight_total_m'])
        for depot, depot_routes in routes.items():
            for route_idx, route in enumerate(depot_routes, 1):
                for seg_idx, (a, b) in enumerate(zip(route[:-1], route[1:]), 1):
                    h_cruise = float(cruise[a][b])
                    ascent = float(ascent_matrix[a][b])
                    descent = float(descent_matrix[a][b])
                    vertical_total = ascent + descent if vertical is None else float(vertical[a][b])
                    writer.writerow([depot, route_idx, seg_idx, a, b, float(heights[a]), float(heights[b]), h_cruise, ascent, float(horizontal[a][b]), descent, vertical_total, float(total[a][b])])
    return output_path

def run_parallel_experiments(random_seeds=CLRP_PARALLEL_RANDOM_SEEDS, max_iter=CLRP_MAX_ITER, no_fly_shp=CLRP_EXPERIMENT_NO_FLY_SHP, administrative_boundary_shp=None, max_workers=CLRP_PARALLEL_MAX_WORKERS, output_root=None):
    seeds = [int(seed) for seed in random_seeds]
    if not seeds:
        raise ValueError('random_seeds must not be empty')
    if len(set(seeds)) != len(seeds):
        raise ValueError('random_seeds contains duplicate seeds')
    problem = _load_problem_once(no_fly_shp, administrative_boundary_shp)
    scenario = problem['scenario_label']
    scenario_file = _sanitize_filename(scenario)
    output_root = os.path.abspath(output_root or os.path.join(_OUTPUT_DIR, _run_output_dirname(scenario, max_iter)))
    os.makedirs(output_root, exist_ok=True)
    fixed_outputs = {'candidate sites': export_candidate_pool_csv(problem, output_root), 'visibility matrix': export_visibility_matrix_csv(problem, output_root), 'flight matrix': export_flight_matrix_checkpoint(problem, output_root), 'node heights': export_node_elevations_csv(problem, output_root), 'unreachable points': export_unplannable_waypoints_csv(problem, output_root)}
    for label, path in fixed_outputs.items():
        if path:
            print(f'{label}: {path}')
    local_blocks = [local for local in (_make_local_block_problem(problem, block) for block in problem['free_flight_blocks']) if local is not None]
    if not local_blocks:
        raise ValueError('no free-flight block contains waypoints')
    task_count = len(seeds) * len(local_blocks)
    workers = max(1, min(int(max_workers), task_count, os.cpu_count() or 1))
    print('\n=== Multi-seed parallel experiment ===')
    print(f'scenario: {scenario}')
    print(f'random seeds: {seeds}')
    print(f'free-flight blocks: {len(local_blocks)}; tasks: {task_count}; workers: {workers}')
    print(f'output directory: {output_root}')
    payloads = [{'problem': local_problem, 'seed': seed, 'max_iter': max_iter, 'output_dir': os.path.join(output_root, f'seed_{seed}')} for seed in seeds for local_problem in local_blocks]
    block_summaries = []
    ctx = mp.get_context('spawn')
    # When loaded via importlib the functions live in a non-standard module name; spawn workers must register it in an initializer.
    pool_kwargs = {}
    if __name__ != '__main__':
        from lib_zunhua_experiment_common import collect_module_config_snapshot, multiprocess_worker_init
        pool_kwargs = {
            'initializer': multiprocess_worker_init,
            'initargs': (os.path.abspath(__file__), __name__, collect_module_config_snapshot(sys.modules[__name__])),
        }
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, **pool_kwargs) as executor:
        future_to_task = {executor.submit(_run_single_seed_worker, payload): (payload['seed'], payload['problem']['block_id']) for payload in payloads}
        for future in as_completed(future_to_task):
            seed, block_id = future_to_task[future]
            try:
                summary = future.result()
                block_summaries.append(summary)
                print(f"[done] seed={seed}, block={block_id + 1}, Best={summary['best_total_cost']}, time={summary['solve_time_seconds']}s")
            except Exception as exc:
                block_summaries.append({'scenario': scenario, 'seed': seed, 'block_id': block_id, 'status': 'failed', 'error': ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))})
                print(f'[failed] seed={seed}, block={block_id + 1}: {exc}')
    seed_summaries = []
    for seed in seeds:
        rows = [row for row in block_summaries if row['seed'] == seed]
        if any((row.get('status') != 'success' for row in rows)):
            seed_summaries.append({'scenario': scenario, 'seed': seed, 'status': 'failed', 'error': 'at least one free-flight block failed to solve'})
            continue
        selected = sorted({depot for row in rows for depot in row['global_selected_depots']})
        routes = {}
        for row in rows:
            for depot, depot_routes in row['global_routes'].items():
                routes.setdefault(int(depot), []).extend(depot_routes)
        build_cost = sum((float(row['build_cost']) for row in rows))
        transport_cost = sum((float(row['transport_cost']) for row in rows))
        total_cost = build_cost + transport_cost
        solve_time = sum((float(row['solve_time_seconds']) for row in rows))
        seed_dir = os.path.join(output_root, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        iteration_csvs = [row['iteration_csv'] for row in rows]
        block_result_files = [row['block_result_json'] for row in rows]
        seed_iterations_csv = os.path.join(seed_dir, f'{_output_stem("iterations_all_blocks", scenario_file, f"seed_{seed}")}.csv')
        _merge_iteration_csvs(rows, seed_iterations_csv)
        solution_result_json = os.path.join(seed_dir, f'{_output_stem("solution_result", scenario_file, f"seed_{seed}")}.json')
        save_solution_result(solution_result_json, selected, routes, build_cost, transport_cost, total_cost, scenario, seed, no_fly_shp=problem['no_fly_shp'], administrative_boundary_shp=problem['administrative_boundary_shp'], solve_time=solve_time, iteration_csvs=iteration_csvs, block_result_files=block_result_files)
        print(f'[result saved] seed={seed}: {solution_result_json}')
        export_route_3d_segments_csv(problem, routes, os.path.join(seed_dir, f'{_output_stem("route_3d_segments", scenario_file, f"seed_{seed}")}.csv'))
        solution_png = os.path.join(seed_dir, f'{_output_stem("solution", scenario_file, f"seed_{seed}")}.png')
        plot_error = ''
        solution_paths = {}
        try:
            solution_paths = show_result_CLRP(problem['customers'], problem['coords'], problem['depot_build_cost'], selected, routes, build_cost, transport_cost, total_cost, problem['prohibited_polygons'], problem['administrative_boundaries'], output_path=solution_png, show_figure=False, scenario_label=scenario, random_seed=seed, unplannable_details=problem['unplannable_details'], navigation_data=problem['navigation_data'], navigation_distances=problem['navigation_distances'])
            print(f'[figures written] seed={seed}: map/legend/route_length')
        except Exception as exc:
            plot_error = f'{type(exc).__name__}: {exc}'
            print(f'[plot failed but result saved] seed={seed}: {plot_error}')
        seed_summaries.append({'scenario': scenario, 'seed': seed, 'status': 'success', 'best_total_cost': f'{total_cost:.10f}', 'build_cost': f'{build_cost:.10f}', 'transport_cost': f'{transport_cost:.10f}', 'n_open_depots': len(selected), 'selected_depots': ';'.join(map(str, selected)), 'n_routes': sum((len(x) for x in routes.values())), 'solution_result_json': solution_result_json, 'solution_png': solution_paths.get('map', '') if not plot_error else '', 'solution_legend_png': solution_paths.get('legend', '') if not plot_error else '', 'solution_route_length_png': solution_paths.get('route_length', '') if not plot_error else '', 'plot_error': plot_error})
    block_summaries.sort(key=lambda row: (int(row['seed']), int(row.get('block_id', -1))))
    seed_summaries.sort(key=lambda row: int(row['seed']))
    block_summary_csv = os.path.join(output_root, f'{_output_stem("block_summary", scenario_file)}.csv')
    summary_csv = os.path.join(output_root, f'{_output_stem("experiment_summary", scenario_file)}.csv')
    all_iterations_csv = os.path.join(output_root, f'{_output_stem("all_iterations", scenario_file)}.csv')
    _write_dict_rows(block_summary_csv, block_summaries)
    _write_dict_rows(summary_csv, seed_summaries)
    _merge_iteration_csvs(block_summaries, all_iterations_csv)
    successful = [row for row in seed_summaries if row.get('status') == 'success']
    statistics_csv = ''
    if successful:
        costs = np.asarray([float(row['best_total_cost']) for row in successful])
        statistics_csv = os.path.join(output_root, f'{_output_stem("experiment_statistics", scenario_file)}.csv')
        _write_dict_rows(statistics_csv, [{'scenario': scenario, 'n_runs': len(successful), 'mean_best_total_cost': f'{costs.mean():.10f}', 'std_best_total_cost': f'{(costs.std(ddof=1) if len(costs) > 1 else 0.0):.10f}', 'min_best_total_cost': f'{costs.min():.10f}', 'max_best_total_cost': f'{costs.max():.10f}', 'best_seed': int(successful[int(np.argmin(costs))]['seed'])}])
    print('\n=== Parallel experiment finished ===')
    print(f'block summary: {block_summary_csv}')
    print(f'per-seed summary: {summary_csv}')
    print(f'all iteration logs: {all_iterations_csv}')
    if statistics_csv:
        print(f'statistics: {statistics_csv}')
    return seed_summaries
if __name__ == '__main__':
    mp.freeze_support()
    if CLRP_REPLOT_RESULT_JSON:
        plot_saved_solution(CLRP_REPLOT_RESULT_JSON)
    else:
        run_parallel_experiments(random_seeds=CLRP_PARALLEL_RANDOM_SEEDS, max_iter=CLRP_MAX_ITER, no_fly_shp=CLRP_EXPERIMENT_NO_FLY_SHP, max_workers=CLRP_PARALLEL_MAX_WORKERS)
