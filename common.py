"""Shared setup: load TLE, compute positions and distances for TSP. Used by all solvers."""
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sgp4.api import Satrec

DATA_PATH = Path(__file__).parent / "leo_debris_sample.csv"
V_KM_S = 5.0  # nominal transfer speed (km/s) for travel-time estimate


def epoch_str_to_jd(epoch_str: str) -> tuple:
    """TLE epoch YYDDD.DDDDDDDD → Julian date (jd, fr)."""
    e = float(epoch_str)
    yy = int(e // 1000)
    doy = e % 1000
    year = 2000 + yy if yy < 57 else 1900 + yy
    base = datetime(year, 1, 1) + timedelta(days=doy - 1)
    jd_total = base.toordinal() + 1721424.5 + (
        base.hour * 3600 + base.minute * 60 + base.second
    ) / 86400.0
    return int(jd_total), jd_total - int(jd_total)


def tle_to_position(line1: str, line2: str, jd: int, fr: float) -> np.ndarray:
    sat = Satrec.twoline2rv(line1, line2)
    e, r, _ = sat.sgp4(jd, fr)
    return np.array(r) if e == 0 else np.full(3, np.nan)


def load_data(data_path: Path = None) -> tuple:
    """
    Load TLE CSV, compute positions and distances for TSP.
    Returns (positions, dist_matrix, dist_depot, targets_df).

    positions: (n, 3) array of target positions in TEME (km)
    dist_matrix: (n, n) Euclidean distance between targets (km)
    dist_depot: (n,) distance from depot to each target (km)
    """
    path = data_path or DATA_PATH
    df = pd.read_csv(path)
    depot_row = df.iloc[[0]]
    targets_df = df.iloc[1:].reset_index(drop=True)

    ref_epoch = depot_row["epoch"].values[0]
    jd, fr = epoch_str_to_jd(str(ref_epoch))
    r_depot = tle_to_position(
        depot_row["line1"].values[0], depot_row["line2"].values[0], jd, fr
    )

    n = len(targets_df)
    positions = np.zeros((n, 3))
    for i in range(n):
        r = tle_to_position(
            targets_df.iloc[i]["line1"], targets_df.iloc[i]["line2"], jd, fr
        )
        positions[i] = r

    # Pairwise distances (km)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            dist_matrix[i, j] = dist_matrix[j, i] = d

    # Depot to each target
    dist_depot = np.array([np.linalg.norm(positions[i] - r_depot) for i in range(n)])

    tle_data = {
        "depot_line1": depot_row["line1"].values[0],
        "depot_line2": depot_row["line2"].values[0],
        "target_line1": targets_df["line1"].tolist(),
        "target_line2": targets_df["line2"].tolist(),
        "jd0": jd,
        "fr0": fr,
    }
    return positions, dist_matrix, dist_depot, targets_df, tle_data


def _position_at_time(tle_data: dict, idx: int, t_days: float) -> np.ndarray:
    """Position of depot (idx=-1) or target idx at epoch + t_days (km, TEME)."""
    jd_total = tle_data["jd0"] + tle_data["fr0"] + t_days
    jd = int(jd_total)
    fr = jd_total - jd
    if idx == -1:
        return tle_to_position(tle_data["depot_line1"], tle_data["depot_line2"], jd, fr)
    return tle_to_position(
        tle_data["target_line1"][idx], tle_data["target_line2"][idx], jd, fr
    )


def tour_distance_time_dependent(
    tour: np.ndarray,
    tle_data: dict,
    v_km_s: float = V_KM_S,
) -> float:
    """
    Total distance simulating motion: propagate satellites forward by travel time
    after each leg. Distance at leg k uses positions at t_k (departure time).
    Travel time = distance / v_km_s; t_{k+1} = t_k + travel_time_days.
    """
    n = len(tour)
    if n == 0:
        return 0.0
    total_km = 0.0
    t_days = 0.0
    sec_per_day = 86400.0

    # Leg 0: depot -> tour[0]
    pos_depot = _position_at_time(tle_data, -1, t_days)
    pos_0 = _position_at_time(tle_data, int(tour[0]), t_days)
    d = float(np.linalg.norm(pos_depot - pos_0))
    total_km += d
    t_days += d / (v_km_s * sec_per_day)

    # Legs 1..n-1: tour[k] -> tour[k+1]
    for k in range(n - 1):
        pos_curr = _position_at_time(tle_data, int(tour[k]), t_days)
        pos_next = _position_at_time(tle_data, int(tour[k + 1]), t_days)
        d = float(np.linalg.norm(pos_curr - pos_next))
        total_km += d
        t_days += d / (v_km_s * sec_per_day)

    # Leg n: tour[-1] -> depot
    pos_last = _position_at_time(tle_data, int(tour[-1]), t_days)
    pos_depot = _position_at_time(tle_data, -1, t_days)
    d = float(np.linalg.norm(pos_last - pos_depot))
    total_km += d

    return total_km


def tour_distance(
    tour: np.ndarray,
    dist_matrix: np.ndarray,
    dist_depot: np.ndarray,
) -> float:
    """Total distance: depot → tour[0] → tour[1] → ... → tour[-1] → depot (km)."""
    if len(tour) == 0:
        return 0.0
    d = dist_depot[tour[0]] + dist_depot[tour[-1]]
    for i in range(len(tour) - 1):
        d += dist_matrix[tour[i], tour[i + 1]]
    return d
