"""Read support heights at two wheel queries from one gravity-aligned point cloud."""
import numpy as np

from tools100.score100 import RADIUS_M


PLANE_TOL_M = 0.002
WHEEL_WIDTH_M = 0.040
MAX_SLOPE = np.tan(np.deg2rad(35.))
XY_EPS_M = 1e-7


def _hull(xy):
    """Counterclockwise boundary of the observed XY support (Andrew's monotone chain)."""
    ordered = sorted(set(map(tuple, xy)))
    if len(ordered) < 3:
        return np.empty((0, 2))

    def chain(points):
        vertices = []
        for p in points:
            while len(vertices) >= 2:
                a, b = vertices[-2:]
                cross = (b[0]-a[0]) * (p[1]-a[1]) - (b[1]-a[1]) * (p[0]-a[0])
                if cross > 1e-12:
                    break
                vertices.pop()
            vertices.append(p)
        return vertices

    return np.asarray(chain(ordered)[:-1] + chain(reversed(ordered))[:-1])


def _inside(poly, xy, margin=-XY_EPS_M):
    if len(poly) < 3:
        return np.zeros(len(xy), dtype=bool)
    edge = np.roll(poly, -1, axis=0) - poly
    offset = xy[:, None, :] - poly
    cross = edge[:, 0] * offset[:, :, 1] - edge[:, 1] * offset[:, :, 0]
    return (cross >= margin * np.linalg.norm(edge, axis=1)).all(axis=1)


def _covers_query(xy):
    return _inside(_hull(xy), np.zeros((1, 2)))[0]


def _has_spread(xy):
    if len(xy) < 3:
        return False
    centered = xy - xy.mean(axis=0)
    return np.linalg.eigvalsh(centered.T @ centered / len(xy))[0] >= PLANE_TOL_M**2


def _horizontal_layers(xy, z):
    """Find measured height bands with two-dimensional support; a row of wall hits has none."""
    remaining = np.sort(z)
    layers = []
    while len(remaining) >= 3:
        ends = np.searchsorted(remaining, remaining + 2*PLANE_TOL_M, side="right")
        counts = ends - np.arange(len(remaining))
        start = int(np.argmax(counts))
        if counts[start] < 3:
            break
        height = float(np.median(remaining[start:ends[start]]))
        band = np.abs(z - height) <= PLANE_TOL_M
        if _has_spread(xy[band]):
            layers.append((float(np.median(z[band])), band))
        remaining = remaining[np.abs(remaining - height) > PLANE_TOL_M]
    return sorted(layers, key=lambda layer: layer[0])


def _riser_segment(wall, wall_z, lower_xy, upper_xy):
    """Return the observed endpoints of a thin riser separating these two supports."""
    if len(wall) < 3 or np.ptp(wall_z) < 2*PLANE_TOL_M:
        return np.empty((0, 2))
    center = wall.mean(axis=0)
    _, _, axes = np.linalg.svd(wall - center, full_matrices=False)
    tangent, normal = axes
    along = (wall - center) @ tangent
    if (np.max(np.abs((wall - center) @ normal)) > PLANE_TOL_M
            or np.ptp(along) < WHEEL_WIDTH_M / 2
            or not along.min() <= -center @ tangent <= along.max()):
        return np.empty((0, 2))
    signed_upper = (upper_xy - center) @ normal
    if np.median(signed_upper) < 0:
        normal = -normal
        signed_upper = -signed_upper
    if signed_upper.min() < -PLANE_TOL_M or not np.any((lower_xy - center) @ normal < -PLANE_TOL_M):
        return np.empty((0, 2))
    return center + np.array([along.min(), along.max()])[:, None] * tangent


def _visible_face(xy, z, low, high):
    between = (z > low[0] + PLANE_TOL_M) & (z < high[0] - PLANE_TOL_M)
    wall, wall_z = xy[between], z[between]
    lower_xy, upper_xy = xy[low[1]], xy[high[1]]
    # A single face can span multiple beams across the wheel. At a corner
    # restrict the fit to the wheel's lane if the wider returns are inconsistent.
    segment = _riser_segment(wall, wall_z, lower_xy, upper_xy)
    if len(segment):
        return segment
    lane = np.abs(wall[:, 1]) <= WHEEL_WIDTH_M
    return _riser_segment(wall[lane], wall_z[lane], lower_xy, upper_xy)


def _plane(xy, z):
    design = np.column_stack((xy, np.ones(len(xy))))
    coeff, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
    return coeff if np.max(np.abs(design @ coeff - z)) <= PLANE_TOL_M else None


def _wheel_height(points, query):
    xy = points[:, :2] - query
    distance = np.linalg.norm(xy, axis=1)
    near = distance <= RADIUS_M
    xy, z, distance = xy[near], points[near, 2], distance[near]
    if not _has_spread(xy):
        return None

    # Allow a small continuation of a locally continuous support. All measured
    # hits within one wheel width must agree, including any visible riser.
    close = distance <= WHEEL_WIDTH_M
    can_continue = distance.min() <= WHEEL_WIDTH_M/2 and close.sum() >= 4 and _has_spread(xy[close])
    coeff = _plane(xy, z)
    if coeff is not None:
        if np.linalg.norm(coeff[:2]) <= MAX_SLOPE and (can_continue or _covers_query(xy)):
            return float(coeff[2])
        return None

    layers = _horizontal_layers(xy, z)
    for height, band in layers:
        if can_continue and band[close].all():
            return height
        # Interpolation entirely inside the distance to a contradictory return
        # cannot bridge a hole between separated patches of the same height.
        radius = distance[~band].min(initial=RADIUS_M)
        if _covers_query(xy[band & (distance < radius - XY_EPS_M)]):
            return height
    if len(layers) != 2 or layers[1][0] - layers[0][0] <= 2*PLANE_TOL_M:
        return None
    low, high = layers
    face = _visible_face(xy, z, low, high)
    top = _hull(np.vstack((xy[high[1]], face)))
    ground = xy[low[1]]
    if _inside(top, ground, margin=PLANE_TOL_M).any():
        return None  # overlapping/disconnected supports cannot be ordered from XY alone
    if _inside(top, np.zeros((1, 2)))[0]:
        return high[0]

    # Only ground evidence on this side of the top footprint can bound the
    # query. The opposite ground surface must not be joined across the ledge.
    edge = np.roll(top, -1, axis=0) - top
    weight = np.clip(-np.sum(top * edge, axis=1) / np.sum(edge * edge, axis=1), 0, 1)
    projected = top + weight[:, None] * edge
    boundary = projected[np.argmin(np.linalg.norm(projected, axis=1))]
    outward = -boundary / np.linalg.norm(boundary)
    support = np.vstack((ground, face))
    outside = (support - boundary) @ outward >= -XY_EPS_M
    return low[0] if _covers_query(support[outside]) else None


def surface_heights_m(points_xyz_m, queries_xy_m):
    """Return each wheel path's height in the input frame, or None if unobserved."""
    points = np.asarray(points_xyz_m, dtype=np.float64)
    if points.shape == (0,):
        points = points.reshape(0, 3)
    queries = np.asarray(queries_xy_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or queries.shape != (2, 2) or not np.isfinite(queries).all():
        raise ValueError("expected N XYZ returns and two finite wheel queries")
    points = points[np.isfinite(points).all(axis=1)]
    return tuple(_wheel_height(points, query) for query in queries)


def surface_delta_mm(points_xyz_m, queries_xy_m):
    """Drop-in comparator for the frozen left-minus-right height contract."""
    left, right = surface_heights_m(points_xyz_m, queries_xy_m)
    return None if left is None or right is None else 1000. * (left - right)
