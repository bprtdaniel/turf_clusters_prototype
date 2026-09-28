"""
turf_clustering.py

This module holds all the functions I wrote for the first version of my
turf-cutting prototype, so I can load them into the main clustering script.

The goal: take a list of households (with coordinates and the number of
target voters living there) and split them into canvassing turfs that stay
under a maximum number of households and a maximum number of voters.

The functions follow the steps of the pipeline:

    Step 1  Data collection       -> build a road distance matrix via OSRM
    Step 2  Quality control       -> check the input data before clustering
    Step 3  Clustering            -> KMeans / KMedoids, with k increasing until
                                     every turf meets the limits
    Step 4  Aggregation           -> summarise households and voters per turf
    Step 5  Visualisation         -> plot the turfs on an interactive map

Expected input: a household-level DataFrame with the columns
    id          unique household ID
    lat, lon    coordinates in decimal degrees
    NUM_VOTERS  number of target voters in the household
"""

import time

import folium
import matplotlib
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import requests
from kmedoids import KMedoids
from sklearn.cluster import KMeans


REQUIRED_COLUMNS = ["id", "lat", "lon", "NUM_VOTERS"]


############################################################################
# Step 1: Data collection - road distance matrix (OSRM)
############################################################################

# Straight-line distance is not how canvassers move, they walk and drive along
# roads. So I wanted real road distances between households, which I get from
# a local OSRM server running in Docker.
#
# The input was too large to send in one request, so I break the matrix down
# into chunks. Logic: take a chunk of origin and a chunk of destination
# households, get their pairwise distances, then loop to the next chunk.
# I landed on 50 as the chunk size, that worked pretty well.

def build_distance_matrix(households, osrm_url="http://localhost:5000/table/v1/driving/",
                          chunk_size=50, pause=0.1, timeout=30):
    """Return an n x n road distance matrix (metres) indexed by household ID."""

    households = households.reset_index(drop=True)  # make sure we have a clean 0-based index

    # OSRM expects coordinates as lon,lat
    coords = list(zip(households["lon"], households["lat"]))
    # keep the IDs so the distances can be mapped back later
    ids = households["id"].tolist()
    n = len(coords)

    results = []

    # loop over the origins in steps of chunk_size
    for orig in range(0, n, chunk_size):

        # make sure we don't miss the last chunk, which may be smaller than chunk_size
        orig_last = min(orig + chunk_size, n)
        origin_coords = coords[orig:orig_last]
        origin_ids = ids[orig:orig_last]

        # now the same for the destinations
        for dest in range(0, n, chunk_size):
            dest_last = min(dest + chunk_size, n)
            dest_coords = coords[dest:dest_last]
            dest_ids = ids[dest:dest_last]

            # build one combined coordinate string, origins first, then destinations
            # OSRM expects them separated by ';'
            all_coords = origin_coords + dest_coords
            coords_str = ";".join(f"{lon},{lat}" for lon, lat in all_coords)

            # tell OSRM which positions in the string are origins and which are destinations
            n_origins = len(origin_coords)
            n_dests = len(dest_coords)
            sources = ";".join(str(x) for x in range(n_origins))
            destinations = ";".join(str(x) for x in range(n_origins, n_origins + n_dests))

            response = requests.get(
                f"{osrm_url}{coords_str}",
                params={
                    "sources": sources,
                    "destinations": destinations,
                    "annotations": "distance",  # distances only, no need for durations
                },
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Earlier I skipped a chunk silently if nothing came back. That leaves
            # gaps in the matrix without anyone noticing, so now it stops loudly instead.
            if data.get("code") != "Ok" or data.get("distances") is None:
                raise RuntimeError(
                    f"OSRM returned no distances for chunk origins {orig}-{orig_last}, "
                    f"destinations {dest}-{dest_last}: {data.get('message', data.get('code'))}"
                )
            distances = data["distances"]

            # Map the results back to the household IDs by index position.
            # This is where I found the major mistake in my first version: the IDs
            # were not mapped correctly, which made the whole matrix unusable.
            for origin_index, origin_id in enumerate(origin_ids):
                for dest_index, dest_id in enumerate(dest_ids):
                    results.append({
                        "origin_id": origin_id,
                        "destination_id": dest_id,
                        # OSRM returns None for unreachable pairs, keep those as NaN
                        "distance_meters": distances[origin_index][dest_index],
                    })

            # short pause between requests, to be gentle with the local server
            time.sleep(pause)

    # long format -> pivot into the actual n x n matrix
    long_df = pd.DataFrame(results)
    matrix = long_df.pivot(index="origin_id", columns="destination_id", values="distance_meters")
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)

    return matrix


############################################################################
# Step 2: Quality control - check the input before clustering
############################################################################

# Bad input quietly produces bad turfs, so I check the household data first
# and stop with a clear message if something is off.

def validate_households(households):
    """Raise a ValueError listing every problem found in the household data."""

    problems = []

    missing = [col for col in REQUIRED_COLUMNS if col not in households.columns]
    if missing:
        # without these columns the other checks make no sense
        raise ValueError(f"Missing required columns: {missing}")

    if households["id"].duplicated().any():
        problems.append(f"{households['id'].duplicated().sum()} duplicate household IDs")

    if households[["lat", "lon"]].isna().any().any():
        problems.append("missing coordinates")

    lat, lon = households["lat"].dropna(), households["lon"].dropna()
    if not lat.between(-90, 90).all() or not lon.between(-180, 180).all():
        problems.append("coordinates outside the valid lat/lon range")

    if (households["NUM_VOTERS"] < 0).any() or households["NUM_VOTERS"].isna().any():
        problems.append("negative or missing voter counts")

    if problems:
        raise ValueError("Household data failed validation: " + "; ".join(problems))

    return True


def validate_distance_matrix(matrix):
    """Check the distance matrix is square, complete and has zeros on the diagonal."""

    problems = []

    if matrix.shape[0] != matrix.shape[1]:
        problems.append(f"matrix is not square ({matrix.shape[0]} x {matrix.shape[1]})")
    elif not (matrix.index == matrix.columns).all():
        problems.append("row and column IDs are not in the same order")

    n_missing = int(matrix.isna().sum().sum())
    if n_missing:
        problems.append(f"{n_missing} missing distances (unreachable pairs or failed requests)")

    if matrix.shape[0] == matrix.shape[1] and not np.allclose(np.diag(matrix.values), 0, equal_nan=True):
        problems.append("diagonal is not zero")

    if problems:
        raise ValueError("Distance matrix failed validation: " + "; ".join(problems))

    return True


def check_num_households(household_dict):
    """Number of households per cluster."""
    return [len(ids) for ids in household_dict.values()]


def check_num_voters(voters_dict):
    """Number of voters per cluster."""
    return [sum(voters) for voters in voters_dict.values()]


def meets_limits(households, max_households, max_voters):
    """True if every cluster stays under both upper limits."""

    household_dict = households.groupby("cluster")["id"].apply(list).to_dict()
    voters_dict = households.groupby("cluster")["NUM_VOTERS"].apply(list).to_dict()

    if max(check_num_households(household_dict)) > max_households:
        return False
    if max(check_num_voters(voters_dict)) > max_voters:
        return False
    return True


############################################################################
# Step 3: Clustering
############################################################################

# 3a. The clustering methods I tested. Each takes a list of [lon, lat]
# coordinates and a number of clusters k, and returns one label per household.

def clustering_kmeans(coords, k):
    # random_state so the results are the same every time I run it
    model = KMeans(n_clusters=k, n_init=10, random_state=0)
    return model.fit_predict(np.array(coords))


def clustering_kmedoids_manhattan(coords, k):
    # Manhattan distance roughly follows a street grid, which a lot of U.S. towns have
    model = KMedoids(n_clusters=k, metric="manhattan", method="pam", random_state=0)
    return model.fit_predict(np.array(coords))


def clustering_kmedoids_haversine(coords, k):
    coords = np.array(coords)
    # Haversine needs coordinates in radians and in [lat, lon] order, so switch the order as well
    coords_rad = np.radians(coords[:, [1, 0]])
    model = KMedoids(n_clusters=k, metric="haversine", method="pam", random_state=0)
    return model.fit_predict(coords_rad)


def clustering_kmedoids_precomputed(distance_matrix, k):
    # input here is the road distance matrix from Step 1, as a numpy array
    model = KMedoids(n_clusters=k, metric="precomputed", method="pam", random_state=0)
    return model.fit(distance_matrix).labels_


CLUSTERING_METHODS = {
    "kmeans": clustering_kmeans,
    "kmedoids_manhattan": clustering_kmedoids_manhattan,
    "kmedoids_haversine": clustering_kmedoids_haversine,
}


# 3b. Iterative clustering
#
# In my first version I had one near-identical function per method. They all
# follow the same logic, so I combined them into one:
# start at k_min, cluster, check the limits, and if any turf is too big,
# increase k by 1 and try again until every turf fits.

def iterative_clustering(households, k_min, k_max, max_households, max_voters, method="kmeans"):
    """Find the smallest k in [k_min, k_max] where every turf meets both limits.

    Returns (households_with_clusters, k), or None if no k in the range works.
    """

    if method not in CLUSTERING_METHODS:
        raise ValueError(f"Unknown method '{method}', choose from {list(CLUSTERING_METHODS)}")

    validate_households(households)
    households = households.copy()  # work on a copy so the original stays untouched
    coords = households[["lon", "lat"]].values.tolist()

    for k in range(k_min, k_max + 1):
        households["cluster"] = CLUSTERING_METHODS[method](coords, k)

        if meets_limits(households, max_households, max_voters):
            print(f"Success with k={k} ({method})")
            return households, k

    print(f"No valid k found in range [{k_min}, {k_max}] ({method})")
    return None


# 3c. KMedoids on real road distances (experimental)
#
# This was the version I was most excited about, clustering on actual road
# distances instead of straight lines. I got clusters back, but they were often
# spread apart: there is no maximum walking distance per turf, so the method can
# satisfy the household and voter limits with turfs that aren't walkable.
# Working through this problem is what led me to re-design the method for
# Turfmaps.app.

def iterative_kmedoids_road(households, distance_matrix, k_min, k_max, max_households, max_voters):
    """Same logic as iterative_clustering, but on the road distance matrix from Step 1."""

    validate_households(households)
    validate_distance_matrix(distance_matrix)

    households = households.copy()
    households["id"] = households["id"].astype(int)
    if "cluster" in households.columns:
        households = households.drop(columns=["cluster"])

    # only keep households that are in the matrix
    matrix_ids = distance_matrix.index.to_numpy().astype(int)
    households = households[households["id"].isin(set(matrix_ids))]

    # OSRM distances are not always symmetric (one-way streets), but KMedoids
    # needs a symmetric matrix, so I take the shorter of the two directions.
    # This is an assumption, but it has to be made.
    D = distance_matrix.to_numpy(dtype=float)
    D = np.minimum(D, D.T)

    for k in range(k_min, k_max + 1):
        labels = clustering_kmedoids_precomputed(D, k)

        # labels come back in matrix order, so map them back onto the household IDs
        assignments = pd.DataFrame({"id": matrix_ids, "cluster": labels})
        clustered = households.merge(assignments, on="id", how="left")

        if meets_limits(clustered, max_households, max_voters):
            print(f"Success with k={k} (kmedoids on road distances)")
            return clustered, k

    print(f"No valid k found in range [{k_min}, {k_max}] (kmedoids on road distances)")
    return None


############################################################################
# Step 4: Aggregation - results per turf
############################################################################

def summarise_clusters(households):
    """One row per turf: number of households, number of voters and the household IDs."""

    summary = (
        households.groupby("cluster")
        .agg(households=("id", "size"), voters=("NUM_VOTERS", "sum"), ids=("id", list))
        .reset_index()
    )
    return summary


# To make the map easier to read, I only show the top 10 turfs.
# I define a top turf as one that holds the most households (and then voters)
# while still meeting the limits.

def get_top_clusters(summary, n=10):
    return summary.sort_values(["households", "voters"], ascending=False).head(n)


############################################################################
# Step 5: Visualisation - interactive map
############################################################################

def plot_clusters_interactive(households, cluster_ids=None, zoom_start=11, tiles="Esri.WorldGrayCanvas"):
    """Plot households as points coloured by turf on a Folium map.

    households:  the clustered household DataFrame from Step 3
    cluster_ids: optional list of turfs to show, e.g. the top 10 from Step 4
    zoom_start:  11 looks good for a city-sized area
    tiles:       background map. I use Esri's light grey canvas: the default
                 OpenStreetMap tiles block maps opened as local files (403
                 "Access blocked"), it needs no API key, and the grey style
                 makes the turf colours easier to see.
    """

    if cluster_ids is not None:
        households = households[households["cluster"].isin(cluster_ids)]

    unique_clusters = sorted(households["cluster"].unique())

    # one colour per turf (colours repeat after 20, which is fine for a top 10)
    colormap = matplotlib.colormaps["tab20"]
    cluster_color = {
        cl: colors.rgb2hex(colormap(i % colormap.N))
        for i, cl in enumerate(unique_clusters)
    }

    # centre the map on the median coordinates
    m = folium.Map(
        location=[households["lat"].median(), households["lon"].median()],
        zoom_start=zoom_start,
        tiles=tiles,
    )

    for _, row in households.iterrows():
        cl = row["cluster"]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            # I only show the turf number here, the ID is no use to the user and the address was too long
            popup=f"Turf: {cl}",
            color=cluster_color[cl],
            fill=True,
            fill_color=cluster_color[cl],
            fill_opacity=0.7,
        ).add_to(m)

    return m
