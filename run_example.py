"""
run_example.py

Runs the whole pipeline on made-up data, so anyone can try it without a
voter file. The households below are random points around Columbus, Ohio,
with a random number of voters each. No real voter data is used anywhere.

Usage:
    python run_example.py

Output:
    output/households_clustered.csv   every household with its turf
    output/turf_summary.csv           households and voters per turf
    output/turf_map.html              interactive map of the top 10 turfs
"""

import os

import numpy as np
import pandas as pd

from turf_clustering import (
    get_top_clusters,
    iterative_clustering,
    plot_clusters_interactive,
    summarise_clusters,
)


# Limits per turf: roughly what one canvasser can cover in a shift
MAX_HOUSEHOLDS = 60
MAX_VOTERS = 100
K_MIN, K_MAX = 2, 60


def make_example_households(n=600, seed=42):
    """Random households in a few neighbourhoods around Columbus, Ohio."""

    rng = np.random.default_rng(seed)
    centres = [(39.99, -83.00), (39.96, -82.97), (40.01, -83.03), (39.95, -83.02)]

    rows = []
    for i in range(n):
        lat, lon = centres[i % len(centres)]
        rows.append({
            "id": i + 1,
            "lat": lat + rng.normal(0, 0.006),
            "lon": lon + rng.normal(0, 0.008),
            "NUM_VOTERS": int(rng.integers(1, 4)),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs("output", exist_ok=True)

    # Step 1 (road distance matrix) needs a local OSRM server, so this example
    # starts with straight-line clustering. See the README for the OSRM setup.
    households = make_example_households()

    # Steps 2 and 3: quality checks run inside iterative_clustering
    result = iterative_clustering(
        households, K_MIN, K_MAX, MAX_HOUSEHOLDS, MAX_VOTERS, method="kmeans"
    )
    if result is None:
        return
    clustered, k = result

    # Step 4: aggregate per turf
    summary = summarise_clusters(clustered)
    top10 = get_top_clusters(summary, n=10)
    print(top10[["cluster", "households", "voters"]].to_string(index=False))

    # Step 5: map the top 10 turfs
    turf_map = plot_clusters_interactive(clustered, cluster_ids=top10["cluster"].tolist())

    # I save everything here, not inside the functions, so the functions stay reusable
    clustered.to_csv("output/households_clustered.csv", index=False)
    summary.drop(columns="ids").to_csv("output/turf_summary.csv", index=False)
    turf_map.save("output/turf_map.html")
    print(f"\n{k} turfs created, files saved to output/")


if __name__ == "__main__":
    main()
