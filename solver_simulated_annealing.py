#!/usr/bin/env python3
"""Simulated annealing TSP: 2-opt with accept worse moves to escape local minima."""
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_data, tour_distance_time_dependent

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def nearest_neighbor_static(dist_matrix: np.ndarray, dist_depot: np.ndarray) -> np.ndarray:
    n = len(dist_depot)
    tour = []
    unvisited = set(range(n))
    current = np.argmin(dist_depot)
    tour.append(current)
    unvisited.remove(current)
    while unvisited:
        best_j = min(unvisited, key=lambda j: dist_matrix[current, j])
        tour.append(best_j)
        unvisited.remove(best_j)
        current = best_j
    return np.array(tour)


def simulated_annealing_tsp(
    tour_init: np.ndarray,
    tle_data: dict,
    T_init: float = 10_000.0,
    T_min: float = 0.1,
    cooling: float = 0.99995,
    max_iters: int = 100_000,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(tour_init)
    tour = tour_init.copy()
    E = tour_distance_time_dependent(tour, tle_data)
    best_tour, best_E = tour.copy(), E
    T = T_init

    for _ in range(max_iters):
        i, j = sorted(rng.integers(0, n, size=2))
        if j - i < 2:
            continue
        new_tour = np.concatenate([
            tour[: i + 1],
            tour[i + 1 : j + 1][::-1],
            tour[j + 1 :],
        ])
        E_new = tour_distance_time_dependent(new_tour, tle_data)
        dE = E_new - E
        if dE <= 0 or rng.random() < np.exp(-dE / T):
            tour, E = new_tour, E_new
            if E < best_E:
                best_tour, best_E = tour.copy(), E
        T = max(T_min, T * cooling)

    return best_tour


def two_opt_improve(tour, tle_data):
    n = len(tour)
    best_dist = tour_distance_time_dependent(tour, tle_data)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                new_tour = np.concatenate([
                    tour[: i + 1], tour[i + 1 : j + 1][::-1], tour[j + 1 :],
                ])
                d = tour_distance_time_dependent(new_tour, tle_data)
                if d < best_dist:
                    tour, best_dist, improved = new_tour, d, True
                    break
            if improved:
                break
    return tour


def main():
    positions, dist_matrix, dist_depot, targets_df, tle_data = load_data()
    n = len(targets_df)
    print(f"Targets: {n} (collect ALL, time-dependent route)")

    tour_init = nearest_neighbor_static(dist_matrix, dist_depot)
    tour_init = two_opt_improve(tour_init, tle_data)
    tour = simulated_annealing_tsp(tour_init, tle_data)
    total_km = tour_distance_time_dependent(tour, tle_data)

    print(f"Simulated annealing (2-opt, time-dependent): total distance = {total_km:.1f} km")

    result_df = pd.DataFrame({
        "order": np.arange(n),
        "index": tour,
        "norad_id": targets_df.iloc[tour]["norad_id"].values,
        "name": targets_df.iloc[tour]["name"].values,
    })
    result_df.to_csv(OUTPUT_DIR / "tour_simulated_annealing.csv", index=False)
    print(f"Saved to {OUTPUT_DIR}/tour_simulated_annealing.csv")


if __name__ == "__main__":
    main()
