#!/usr/bin/env python3
"""Greedy TSP: random initial path + 2-opt local search (only accept improving swaps)."""
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_data, tour_distance_time_dependent

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def random_tour(n: int, seed: int = 42) -> np.ndarray:
    """Random permutation as initial tour (no purposeful ordering)."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


def two_opt_improve(
    tour: np.ndarray,
    tle_data: dict,
) -> np.ndarray:
    """Apply 2-opt moves until no improvement (greedy). Uses time-dependent distance."""
    n = len(tour)
    best_dist = tour_distance_time_dependent(tour, tle_data)

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                new_tour = np.concatenate([
                    tour[: i + 1],
                    tour[i + 1 : j + 1][::-1],
                    tour[j + 1 :],
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
    print(f"Targets: {n} (collect ALL, optimize route)")

    tour = random_tour(n)
    tour = two_opt_improve(tour, tle_data)
    total_km = tour_distance_time_dependent(tour, tle_data)

    print(f"Greedy (random path + 2-opt, time-dependent): total distance = {total_km:.1f} km")

    result_df = pd.DataFrame({
        "order": np.arange(n),
        "index": tour,
        "norad_id": targets_df.iloc[tour]["norad_id"].values,
        "name": targets_df.iloc[tour]["name"].values,
    })
    result_df.to_csv(OUTPUT_DIR / "tour_greedy.csv", index=False)
    print(f"Saved to {OUTPUT_DIR}/tour_greedy.csv")


if __name__ == "__main__":
    main()
