#!/usr/bin/env python3
"""Quantum annealing: TSP via classical SA (neal) – TSP QUBO encoding is complex."""
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


def sa_tsp_classical(
    tour_init: np.ndarray,
    tle_data: dict,
    num_reads: int = 100,
    iters_per_read: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """TSP via simulated annealing – time-dependent distance."""
    rng = np.random.default_rng(seed)
    n = len(tour_init)
    best_tour, best_E = tour_init.copy(), tour_distance_time_dependent(tour_init, tle_data)

    for _ in range(num_reads):
        tour = rng.permutation(n)
        E = tour_distance_time_dependent(tour, tle_data)
        T = 5000.0
        for _ in range(iters_per_read):
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
            T = max(0.1, T * 0.9999)

    return best_tour


def main():
    try:
        from dwave.system import DWaveSampler, EmbeddingComposite
        print("D-Wave TSP encoding not implemented – using classical SA")
    except Exception:
        pass

    positions, dist_matrix, dist_depot, targets_df, tle_data = load_data()
    n = len(targets_df)
    print(f"Targets: {n} (collect ALL, time-dependent route)")

    tour_init = nearest_neighbor_static(dist_matrix, dist_depot)
    tour = sa_tsp_classical(tour_init, tle_data)
    total_km = tour_distance_time_dependent(tour, tle_data)

    print(f"Classical SA (TSP, time-dependent): total distance = {total_km:.1f} km")

    result_df = pd.DataFrame({
        "order": np.arange(n),
        "index": tour,
        "norad_id": targets_df.iloc[tour]["norad_id"].values,
        "name": targets_df.iloc[tour]["name"].values,
    })
    result_df.to_csv(OUTPUT_DIR / "tour_quantum.csv", index=False)
    print(f"Saved to {OUTPUT_DIR}/tour_quantum.csv")


if __name__ == "__main__":
    main()
