"""大规模禁飞区安全绕行距离与航迹恢复。

设计目标
--------
1. 保留“直线优先；受阻时采用近切线接入—安全边界绕行—近切线切出”的逻辑；
2. 边界采样点只用于单航段绕行，不进入 ALNS 服务节点序列；
4. 支持按自由飞行连通分块只计算可能使用的节点对；
5. 对单一闭合禁飞边界使用累计弧长直接求边界最短路，不构造边界全距离矩阵。
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Iterable

import numpy as np
import shapely
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString
from shapely.ops import unary_union


_FLOAT_DTYPE = np.float32
_INT_DTYPE = np.int32


def _parts(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    return []


def _blocked_mask(lines, safe_area):
    """进入安全排斥区内部才判为阻挡；仅切触边界允许通过。"""
    intersects = shapely.intersects(lines, safe_area)
    touches = shapely.touches(lines, safe_area)
    return np.asarray(intersects & ~touches, dtype=bool)


def _line_is_clear(start_xy, end_xy, safe_area) -> bool:
    line = LineString([start_xy, end_xy])
    return not bool(line.intersects(safe_area) and not line.touches(safe_area))


def _sample_ring_preserve_vertices(ring, max_segment_length: float):
    """保留原边界全部顶点，并仅在原线段内部加密。

    相邻采样点始终落在同一条原始边界线段上，因此不会像等弧长插值后再连线
    那样跨越拐点并切入安全区。
    """
    raw = np.asarray(ring.coords, dtype=float)
    if len(raw) < 4:
        return np.empty((0, 2), dtype=float)
    raw = raw[:-1]  # 去掉闭合重复首点
    step = max(float(max_segment_length), 0.5)
    sampled = []
    count = len(raw)
    for i in range(count):
        p0 = raw[i]
        p1 = raw[(i + 1) % count]
        if not sampled or np.linalg.norm(p0 - sampled[-1]) > 1e-9:
            sampled.append(p0)
        length = float(np.linalg.norm(p1 - p0))
        if length <= step:
            continue
        pieces = int(math.ceil(length / step))
        for k in range(1, pieces):
            sampled.append(p0 + (k / pieces) * (p1 - p0))
    result = np.asarray(sampled, dtype=float)
    if len(result) > 1:
        keep = np.r_[True, np.linalg.norm(np.diff(result, axis=0), axis=1) > 1e-9]
        result = result[keep]
    return result


def _build_boundary_data(navigation_safe, requested_step: float, max_boundary_nodes: int):
    polygons = _parts(navigation_safe)
    if not polygons:
        return np.empty((0, 2), dtype=float), [], [], float(requested_step)

    total_perimeter = float(sum(poly.exterior.length for poly in polygons))
    max_boundary_nodes = max(128, int(max_boundary_nodes))
    effective_step = max(float(requested_step), total_perimeter / max_boundary_nodes)

    boundary_chunks = []
    ring_ranges = []
    ring_metrics = []
    cursor = 0
    for ring_id, polygon in enumerate(polygons):
        xy = _sample_ring_preserve_vertices(polygon.exterior, effective_step)
        if len(xy) < 3:
            continue
        start = cursor
        end = start + len(xy)
        boundary_chunks.append(xy)
        ring_ranges.append((start, end))

        next_xy = np.roll(xy, -1, axis=0)
        edge_lengths = np.linalg.norm(next_xy - xy, axis=1)
        cumulative = np.zeros(len(xy), dtype=float)
        if len(xy) > 1:
            cumulative[1:] = np.cumsum(edge_lengths[:-1])
        ring_metrics.append(
            {
                "ring_id": ring_id,
                "start": start,
                "end": end,
                "cumulative": cumulative,
                "edge_lengths": edge_lengths,
                "length": float(edge_lengths.sum()),
            }
        )
        cursor = end

    if not boundary_chunks:
        return np.empty((0, 2), dtype=float), [], [], effective_step
    return np.vstack(boundary_chunks), ring_ranges, ring_metrics, effective_step


def _angular_extrema_candidates(point_xy, boundary_xy, ring_ranges, neighbour_span: int = 2):
    """用极角局部极值近似端点对边界的切点候选。

    该步骤只执行 NumPy 运算，不为端点到每个边界点创建 Shapely 线对象。
    """
    p = np.asarray(point_xy, dtype=float)
    candidates = set()
    for start, end in ring_ranges:
        ring = boundary_xy[start:end]
        m = len(ring)
        if m < 3:
            continue
        vectors = ring - p
        angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
        delta = np.diff(angles)
        signs = np.sign(delta)
        if len(signs):
            # 填充零斜率，避免平直边界遗漏极值。
            for i in range(1, len(signs)):
                if signs[i] == 0:
                    signs[i] = signs[i - 1]
            for i in range(len(signs) - 2, -1, -1):
                if signs[i] == 0:
                    signs[i] = signs[i + 1]
        extrema = {int(np.argmin(angles)), int(np.argmax(angles))}
        changes = np.flatnonzero(signs[:-1] * signs[1:] < 0) + 1 if len(signs) > 1 else []
        extrema.update(int(i) for i in changes)
        for local in extrema:
            for offset in range(-neighbour_span, neighbour_span + 1):
                candidates.add(start + ((local + offset) % m))
    return candidates


def _select_terminal_access_points(
    point_xy,
    boundary_xy,
    ring_ranges,
    safe_area,
    boundary_tree,
    max_count: int,
    candidate_limit: int,
):
    """从近切点候选和最近边界候选中筛选安全接入点。"""
    max_count = max(4, int(max_count))
    candidate_limit = max(max_count * 4, int(candidate_limit))
    m = len(boundary_xy)
    nearest_count = min(m, max(24, max_count * 6))
    _, nearest = boundary_tree.query(np.asarray(point_xy, dtype=float), k=nearest_count)
    nearest = np.atleast_1d(nearest).astype(int)

    tangent_like = _angular_extrema_candidates(point_xy, boundary_xy, ring_ranges)
    distances = np.linalg.norm(boundary_xy - np.asarray(point_xy, dtype=float), axis=1)
    ordered_tangent = sorted(tangent_like, key=lambda idx: (distances[idx], idx))

    ordered = []
    seen = set()
    for idx in ordered_tangent + nearest.tolist():
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        ordered.append(idx)
        if len(ordered) >= candidate_limit:
            break
    if not ordered:
        return np.empty(0, dtype=_INT_DTYPE), np.empty(0, dtype=_FLOAT_DTYPE)

    candidate_xy = boundary_xy[np.asarray(ordered, dtype=int)]
    starts = np.repeat(np.asarray(point_xy, dtype=float)[None, :], len(candidate_xy), axis=0)
    lines = shapely.linestrings(np.stack([starts, candidate_xy], axis=1))
    clear = ~_blocked_mask(lines, safe_area)
    clear_indices = [ordered[i] for i in np.flatnonzero(clear)]
    if not clear_indices:
        return np.empty(0, dtype=_INT_DTYPE), np.empty(0, dtype=_FLOAT_DTYPE)

    tangent_set = set(ordered_tangent)
    clear_indices.sort(key=lambda idx: (0 if idx in tangent_set else 1, distances[idx], idx))
    selected = np.asarray(clear_indices[:max_count], dtype=_INT_DTYPE)
    return selected, distances[selected].astype(_FLOAT_DTYPE)


def _boundary_ring_lookup(ring_metrics, n_boundary):
    ring_id = np.full(n_boundary, -1, dtype=_INT_DTYPE)
    ring_position = np.full(n_boundary, -1, dtype=_INT_DTYPE)
    boundary_s = np.zeros(n_boundary, dtype=float)
    for metric_index, metric in enumerate(ring_metrics):
        start, end = metric["start"], metric["end"]
        ring_id[start:end] = metric_index
        ring_position[start:end] = np.arange(end - start, dtype=_INT_DTYPE)
        boundary_s[start:end] = metric["cumulative"]
    return ring_id, ring_position, boundary_s


def _build_multi_ring_graph(boundary_xy, ring_metrics, safe_area, boundary_link_neighbors):
    edges = {}

    def add_edge(a, b):
        if a == b:
            return False
        if not _line_is_clear(boundary_xy[a], boundary_xy[b], safe_area):
            return False
        value = float(np.linalg.norm(boundary_xy[a] - boundary_xy[b]))
        edges[(a, b)] = min(value, edges.get((a, b), np.inf))
        edges[(b, a)] = min(value, edges.get((b, a), np.inf))
        return True

    for metric in ring_metrics:
        start, end = metric["start"], metric["end"]
        for idx in range(start, end):
            nxt = start if idx + 1 == end else idx + 1
            add_edge(idx, nxt)

    for ring_a, metric_a in enumerate(ring_metrics):
        source = boundary_xy[metric_a["start"] : metric_a["end"]]
        for metric_b in ring_metrics[ring_a + 1 :]:
            target = boundary_xy[metric_b["start"] : metric_b["end"]]
            if not len(source) or not len(target):
                continue
            tree = cKDTree(target)
            k = min(12, len(target))
            distances, nearby = tree.query(source, k=k)
            candidates = []
            for local_a, (dists, neighbours) in enumerate(
                zip(np.atleast_2d(distances), np.atleast_2d(nearby))
            ):
                for distance, local_b in zip(np.atleast_1d(dists), np.atleast_1d(neighbours)):
                    candidates.append(
                        (
                            float(distance),
                            metric_a["start"] + local_a,
                            metric_b["start"] + int(local_b),
                        )
                    )
            found = 0
            for _, a, b in sorted(candidates)[:384]:
                if add_edge(a, b):
                    found += 1
                if found >= int(boundary_link_neighbors):
                    break

    rows = [a for a, _ in edges]
    cols = [b for _, b in edges]
    weights = [edges[key] for key in edges]
    return csr_matrix((weights, (rows, cols)), shape=(len(boundary_xy), len(boundary_xy)))


def _single_ring_base_distances(source_nodes, destination_nodes, boundary_s, ring_length):
    delta = np.abs(
        boundary_s[source_nodes][:, None, None]
        - boundary_s[destination_nodes][None, :, :]
    )
    return np.minimum(delta, ring_length - delta)


def _compute_detours_single_ring(
    source_access,
    source_lengths,
    target_access,
    target_lengths,
    boundary_s,
    ring_length,
):
    base = _single_ring_base_distances(source_access, target_access, boundary_s, ring_length)
    candidate = source_lengths[:, None, None] + base + target_lengths[None, :, :]
    return np.min(candidate, axis=(0, 2))


def _pair_lines_visibility(source_xy, target_xy, safe_area):
    starts = np.repeat(np.asarray(source_xy, dtype=float)[None, :], len(target_xy), axis=0)
    lines = shapely.linestrings(np.stack([starts, target_xy], axis=1))
    return ~_blocked_mask(lines, safe_area)


def build_navigation_distances(
    coords,
    prohibited_polygons,
    clearance: float = 100.0,
    boundary_step: float = 50.0,
    boundary_neighbors: int = 8,
    boundary_link_neighbors: int = 2,
    navigation_margin: float = 1.0,
    component_ids=None,
    max_edge_length: float | None = None,
    max_boundary_nodes: int = 6000,
    terminal_candidate_limit: int = 64,
    target_chunk_size: int = 256,
):
    """构建大规模节点对安全绕行距离矩阵。

    `component_ids` 用于自由飞行分块。不同分块节点对直接置为不可达，避免对
    本来不会进入同一条航线的节点对执行相交判断和绕行计算。
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords 必须是 n×2 坐标数组")
    coords = coords[:, :2]
    n = len(coords)
    if component_ids is None:
        component_ids = np.zeros(n, dtype=_INT_DTYPE)
    else:
        component_ids = np.asarray(component_ids, dtype=_INT_DTYPE)
        if len(component_ids) != n:
            raise ValueError("component_ids 长度必须与 coords 一致")

    result = np.full((n, n), np.inf, dtype=_FLOAT_DTYPE)
    direct_visible = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(result, 0.0)
    np.fill_diagonal(direct_visible, True)

    valid_polygons = [p for p in prohibited_polygons or [] if p is not None and not p.is_empty]
    if not valid_polygons:
        for component in np.unique(component_ids[component_ids >= 0]):
            indices = np.flatnonzero(component_ids == component)
            local = coords[indices]
            distances = np.linalg.norm(local[:, None, :] - local[None, :, :], axis=2)
            if max_edge_length is not None:
                distances[distances > float(max_edge_length)] = np.inf
            result[np.ix_(indices, indices)] = distances.astype(_FLOAT_DTYPE)
            direct_visible[np.ix_(indices, indices)] = np.isfinite(distances)
        return result, None

    original = unary_union(valid_polygons)
    required_safe = original.buffer(float(clearance))
    navigation_safe = required_safe.buffer(max(float(navigation_margin), 0.05))
    if required_safe.is_empty or navigation_safe.is_empty:
        raise ValueError("禁飞区安全排斥面为空，无法构建绕行矩阵")
    shapely.prepare(required_safe)

    boundary_xy, ring_ranges, ring_metrics, effective_step = _build_boundary_data(
        navigation_safe,
        requested_step=float(boundary_step),
        max_boundary_nodes=int(max_boundary_nodes),
    )
    if not len(boundary_xy):
        raise ValueError("未能从安全排斥面构建有效导航边界")

    boundary_tree = cKDTree(boundary_xy)
    access_count = min(max(4, int(boundary_neighbors)), len(boundary_xy))
    access_indices = np.full((n, access_count), -1, dtype=_INT_DTYPE)
    access_lengths = np.full((n, access_count), np.inf, dtype=_FLOAT_DTYPE)

    print(
        f"[绕行加速] 服务节点={n}，边界节点={len(boundary_xy)}，"
        f"有效边界步长≈{effective_step:.1f}m，接入候选={access_count}"
    )
    for terminal, point_xy in enumerate(coords):
        selected, lengths = _select_terminal_access_points(
            point_xy,
            boundary_xy,
            ring_ranges,
            required_safe,
            boundary_tree,
            max_count=access_count,
            candidate_limit=terminal_candidate_limit,
        )
        count = len(selected)
        if count:
            access_indices[terminal, :count] = selected
            access_lengths[terminal, :count] = lengths
        if terminal and terminal % 500 == 0:
            print(f"[绕行加速] 已建立端点接入候选 {terminal}/{n}")

    ring_id, ring_position, boundary_s = _boundary_ring_lookup(ring_metrics, len(boundary_xy))
    single_ring = len(ring_metrics) == 1
    boundary_graph = None
    source_nodes = None
    source_distances = None
    source_row = None

    if not single_ring:
        boundary_graph = _build_multi_ring_graph(
            boundary_xy,
            ring_metrics,
            required_safe,
            boundary_link_neighbors=boundary_link_neighbors,
        )
        unique_access = np.unique(access_indices[access_indices >= 0]).astype(_INT_DTYPE)
        source_nodes = unique_access
        source_row = {int(node): row for row, node in enumerate(unique_access)}
        print(
            f"[绕行加速] 多边界模式：仅从 {len(unique_access)} 个实际接入节点执行 Dijkstra，"
            f"不再计算全部 {len(boundary_xy)}×{len(boundary_xy)} 边界距离。"
        )
        source_distances = dijkstra(
            boundary_graph,
            directed=False,
            indices=unique_access,
            return_predecessors=False,
        ).astype(_FLOAT_DTYPE)

    components = [int(x) for x in np.unique(component_ids) if int(x) >= 0]
    for comp_order, component in enumerate(components, 1):
        indices = np.flatnonzero(component_ids == component)
        if len(indices) == 0:
            continue
        print(
            f"[绕行加速] 连通分块 {component} ({comp_order}/{len(components)})：节点={len(indices)}"
        )
        for local_pos, a in enumerate(indices[:-1]):
            targets = indices[local_pos + 1 :]
            vectors = coords[targets] - coords[a]
            euclidean = np.linalg.norm(vectors, axis=1)
            if max_edge_length is not None:
                usable = euclidean <= float(max_edge_length) + 1e-9
                targets = targets[usable]
                euclidean = euclidean[usable]
            if not len(targets):
                continue

            for chunk_start in range(0, len(targets), max(16, int(target_chunk_size))):
                chunk_targets = targets[chunk_start : chunk_start + target_chunk_size]
                chunk_euclidean = euclidean[chunk_start : chunk_start + target_chunk_size]
                clear = _pair_lines_visibility(coords[a], coords[chunk_targets], required_safe)
                clear_targets = chunk_targets[clear]
                clear_dist = chunk_euclidean[clear]
                if len(clear_targets):
                    result[a, clear_targets] = clear_dist.astype(_FLOAT_DTYPE)
                    result[clear_targets, a] = clear_dist.astype(_FLOAT_DTYPE)
                    direct_visible[a, clear_targets] = True
                    direct_visible[clear_targets, a] = True

                blocked_targets = chunk_targets[~clear]
                if not len(blocked_targets):
                    continue
                source_slots = np.flatnonzero(access_indices[a] >= 0)
                if not len(source_slots):
                    continue
                source_access = access_indices[a, source_slots]
                source_len = access_lengths[a, source_slots].astype(float)
                target_access = access_indices[blocked_targets]
                target_len = access_lengths[blocked_targets].astype(float)

                if single_ring:
                    detour = _compute_detours_single_ring(
                        source_access,
                        source_len,
                        target_access,
                        target_len,
                        boundary_s,
                        ring_metrics[0]["length"],
                    )
                else:
                    valid_dest = target_access >= 0
                    safe_dest = target_access.copy()
                    safe_dest[~valid_dest] = 0
                    src_rows = np.asarray([source_row[int(x)] for x in source_access], dtype=int)
                    base = source_distances[src_rows[:, None, None], safe_dest[None, :, :]]
                    base[:, ~valid_dest] = np.inf
                    candidate = source_len[:, None, None] + base + target_len[None, :, :]
                    detour = np.min(candidate, axis=(0, 2))

                if max_edge_length is not None:
                    detour = np.where(detour <= float(max_edge_length) + 1e-9, detour, np.inf)
                result[a, blocked_targets] = detour.astype(_FLOAT_DTYPE)
                result[blocked_targets, a] = detour.astype(_FLOAT_DTYPE)

            if local_pos and local_pos % 500 == 0:
                print(
                    f"[绕行加速] 分块 {component} 节点对进度 {local_pos}/{len(indices)}"
                )

    return result, {
        "format": "scalable_v1",
        "terminal_xy": coords,
        "component_ids": component_ids,
        "boundary_xy": boundary_xy,
        "ring_ranges": ring_ranges,
        "ring_metrics": ring_metrics,
        "ring_id": ring_id,
        "ring_position": ring_position,
        "boundary_s": boundary_s,
        "direct_visible": direct_visible,
        "distance_matrix": result,
        "access_indices": access_indices,
        "access_lengths": access_lengths,
        "safe_boundary": required_safe,
        "navigation_boundary": navigation_safe,
        "single_ring": single_ring,
        "boundary_graph": boundary_graph,
        "boundary_source_nodes": source_nodes,
        "boundary_source_distances": source_distances,
        "requested_boundary_step": float(boundary_step),
        "effective_boundary_step": float(effective_step),
        "boundary_neighbors": int(access_count),
        "navigation_margin": float(navigation_margin),
        "path_cache": OrderedDict(),
    }


def _single_ring_chain(metric, source_node: int, destination_node: int):
    start, end = int(metric["start"]), int(metric["end"])
    count = end - start
    src = source_node - start
    dst = destination_node - start
    if not (0 <= src < count and 0 <= dst < count):
        return None
    cumulative = metric["cumulative"]
    total = float(metric["length"])
    forward_distance = float((cumulative[dst] - cumulative[src]) % total)
    backward_distance = total - forward_distance
    if forward_distance <= backward_distance:
        local = [src]
        cursor = src
        while cursor != dst:
            cursor = (cursor + 1) % count
            local.append(cursor)
    else:
        local = [src]
        cursor = src
        while cursor != dst:
            cursor = (cursor - 1) % count
            local.append(cursor)
    return np.asarray([start + i for i in local], dtype=int)


def _graph_chain(navigation_data, source_node: int, destination_node: int):
    cache = navigation_data.setdefault("path_cache", OrderedDict())
    key = (int(source_node), int(destination_node))
    if key in cache:
        chain = cache.pop(key)
        cache[key] = chain
        return chain.copy()

    graph = navigation_data.get("boundary_graph")
    if graph is None:
        return None
    _, predecessors = dijkstra(
        graph,
        directed=False,
        indices=int(source_node),
        return_predecessors=True,
    )
    cursor = int(destination_node)
    chain = [cursor]
    while cursor != int(source_node):
        cursor = int(predecessors[cursor])
        if cursor < 0:
            return None
        chain.append(cursor)
    chain.reverse()
    chain = np.asarray(chain, dtype=int)
    cache[key] = chain
    while len(cache) > 256:
        cache.popitem(last=False)
    return chain.copy()


def navigation_leg_coordinates(navigation_data, start_terminal, end_terminal):
    """还原单个安全航段。"""
    if navigation_data is None:
        return None
    start_terminal = int(start_terminal)
    end_terminal = int(end_terminal)
    terminals = navigation_data["terminal_xy"]

    if navigation_data["direct_visible"][start_terminal, end_terminal]:
        return terminals[[start_terminal, end_terminal]]
    if not np.isfinite(navigation_data["distance_matrix"][start_terminal, end_terminal]):
        return None

    access_indices = navigation_data["access_indices"]
    access_lengths = navigation_data["access_lengths"]
    source_slots = np.flatnonzero(access_indices[start_terminal] >= 0)
    destination_slots = np.flatnonzero(access_indices[end_terminal] >= 0)
    if not len(source_slots) or not len(destination_slots):
        return None

    source_nodes = access_indices[start_terminal, source_slots]
    destination_nodes = access_indices[end_terminal, destination_slots]
    source_len = access_lengths[start_terminal, source_slots].astype(float)
    destination_len = access_lengths[end_terminal, destination_slots].astype(float)

    if navigation_data.get("single_ring"):
        boundary_s = navigation_data["boundary_s"]
        total = float(navigation_data["ring_metrics"][0]["length"])
        base = _single_ring_base_distances(source_nodes, destination_nodes[None, :], boundary_s, total)
        base = base[:, 0, :]
    else:
        source_lookup = {
            int(node): row
            for row, node in enumerate(navigation_data["boundary_source_nodes"])
        }
        rows = np.asarray([source_lookup[int(node)] for node in source_nodes], dtype=int)
        base = navigation_data["boundary_source_distances"][
            rows[:, None], destination_nodes[None, :]
        ]

    values = source_len[:, None] + base + destination_len[None, :]
    source_choice, destination_choice = np.unravel_index(np.argmin(values), values.shape)
    if not np.isfinite(values[source_choice, destination_choice]):
        return None
    source_boundary = int(source_nodes[source_choice])
    destination_boundary = int(destination_nodes[destination_choice])

    if navigation_data.get("single_ring"):
        chain = _single_ring_chain(
            navigation_data["ring_metrics"][0], source_boundary, destination_boundary
        )
    else:
        chain = _graph_chain(navigation_data, source_boundary, destination_boundary)
    if chain is None:
        return None

    return np.vstack(
        (
            terminals[start_terminal],
            navigation_data["boundary_xy"][chain],
            terminals[end_terminal],
        )
    )


def route_navigation_coordinates(navigation_data, route):
    """将服务节点序列展开为实际安全航迹。"""
    if navigation_data is None:
        return None
    pieces = []
    for a, b in zip(route[:-1], route[1:]):
        leg = navigation_leg_coordinates(navigation_data, a, b)
        if leg is None:
            return None
        pieces.append(leg if not pieces else leg[1:])
    return np.vstack(pieces) if pieces else np.empty((0, 2), dtype=float)
