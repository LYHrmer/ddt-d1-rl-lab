"""Query one older packet at today's wheel locations after a single-packet abstention."""
import numpy as np

from tools104.motion104 import capture_query
from tools105.surface105 import surface_delta_mm


def after_abstention(current_mm, queries, previous_points, motion):
    """Return a height difference in mm. Motion maps current capture XY to previous capture XY."""
    if current_mm is not None:
        return current_mm
    if previous_points is None or motion is None:
        return None
    return surface_delta_mm(previous_points, capture_query(queries, motion))


def packet_readouts(points, queries, previous, motions):
    """Use the two returns for a wheel pair from the same packet, including for historical reads."""
    current = np.full(len(points), np.nan)
    history = np.full_like(current, np.nan)
    for i, j in enumerate(previous):
        if j != -1 and j != i - 1:
            raise ValueError("expected the immediately preceding packet of the same stream")
        value = surface_delta_mm(points[i], queries[i])
        if value is not None:
            current[i] = value
        result = after_abstention(value, queries[i], points[j] if j >= 0 else None, motions[i])
        if result is not None:
            history[i] = result
    return current, history
