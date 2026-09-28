# Turf Clustering Prototype

(this repo was created from scratch for a more cleaned up version of a different, private repo)
This is the prototype of a larger project: automating **turf cutting** for election campaigns.
This was developed months ago but was the starting point for the methodology of the overarching project.
This is a cleaned-up version of the prototype I wrote in 2025, reorganised and documented for sharing.
The feedback and insights that were obtained through this prototype were used to create turfmaps.app, with a revised approach and better quality outputs.

## The problem

Before a campaign sends volunteers out to knock on doors, someone has to split each precinct into "turfs": small areas that one canvasser can cover in a shift. On the campaigns I worked on, staff drew these by hand, precinct by precinct, taking 5–15 minutes each. Across a whole state, that adds up to hundreds of staff hours in the busiest weeks of the campaign.

A good turf needs to:

- stay under a maximum number of **households** (doors to knock),
- stay under a maximum number of **target voters**,
- and be **walkable**, meaning the households are close together along real roads.

## What this prototype does

The pipeline takes a household-level dataset (coordinates and the number of target voters per household) and runs it through five steps:

| Step | What happens | Where |
|---|---|---|
| 1. Data collection | Builds a road distance matrix between all households from a local [OSRM](https://project-osrm.org/) server, in chunks of 50 × 50 | `build_distance_matrix` |
| 2. Quality control | Checks the input before clustering: required columns, duplicate IDs, missing or invalid coordinates, negative voter counts, and a complete, square distance matrix | `validate_households`, `validate_distance_matrix` |
| 3. Clustering | Clusters households with KMeans or KMedoids, increasing *k* until every turf meets both limits | `iterative_clustering`, `iterative_kmedoids_road` |
| 4. Aggregation | Summarises households and voters per turf and picks the top 10 | `summarise_clusters`, `get_top_clusters` |
| 5. Visualisation | Plots the turfs on an interactive Folium map | `plot_clusters_interactive` |

I tested four ways of measuring distance between households:

- **KMeans** (straight-line, Euclidean) as a baseline
- **KMedoids with Manhattan distance**, which roughly follows a street grid
- **KMedoids with Haversine distance**, which accounts for the Earth's curvature
- **KMedoids on real road distances** from OSRM

## What I learned

The road-distance version was the one I was most excited about, and also the one that taught me the most. It returned clusters, but they were often spread apart: the method has no limit on how far a canvasser has to walk, so it can meet the household and voter limits with turfs that aren't actually walkable.

Getting the distance matrix right was also harder than expected. In my first attempt, the distances weren't mapped back to the correct household IDs, which made the whole matrix unusable. That's why the matrix is now validated before it's used.

Working through these limits is what led me to rethink the problem entirely and re-design the method. That approach became [**Turfmaps.app**](https://turfmaps.app/), which I've built out and maintain on my own.

## Try it yourself

```bash
pip install -r requirements.txt
python run_example.py
```

The example uses **made-up households** around Columbus, Ohio, so it runs without any voter data. It writes three files to `output/`:

- `households_clustered.csv`, every household with its turf
- `turf_summary.csv`, households and voters per turf
- `turf_map.html`, an interactive map of the top 10 turfs

### Using real road distances

Step 1 needs a local OSRM server. I ran mine in Docker with an Ohio extract from [Geofabrik](https://download.geofabrik.de/north-america/us/ohio.html), following the [OSRM Docker instructions](https://github.com/Project-OSRM/osrm-backend#using-docker). Once it's running on `localhost:5000`:

```python
from turf_clustering import build_distance_matrix, iterative_kmedoids_road

matrix = build_distance_matrix(households)
result = iterative_kmedoids_road(households, matrix, k_min=2, k_max=60,
                                 max_households=60, max_voters=100)
```

## Data and privacy

This repository contains **no voter data**. Real voter files include personal information, so they never go into version control. The `.gitignore` excludes all CSV and output files as a safeguard.

## Files

```
turf_clustering.py   all pipeline functions, organised by step
run_example.py       runs the full pipeline on made-up data
requirements.txt     Python dependencies
```
