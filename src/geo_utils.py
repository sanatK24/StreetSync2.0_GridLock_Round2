"""
geo_utils.py
============
Four spatial capabilities, all built only from the dataset itself plus one
free, keyless public API (no paid geocoding/maps service required):

1. nearest_context()   - given any lat/lon (e.g. a point a user clicks on the
   map for a brand-new planned event), find the nearest historical corridor /
   police_station / zone, even if that exact spot was never reported before.

2. similar_incidents()  - the "post-event learning" retrieval engine. Pulls
   the K most similar past events (same cause, nearby location) and returns
   what ACTUALLY happened to them (real durations, real closure outcomes).
   This is deliberately evidence-based rather than a single opaque model
   number - it is what the dashboard shows under "Similar past incidents"
   and is the most defensible part of the duration story, given the ML
   duration regressor's modest accuracy (documented honestly in
   models/metrics.json).

3. get_diversion_suggestion() - tries live OpenStreetMap routing via OSMnx
   to propose a real alternate route around a closure point. This requires
   internet access (free, no API key - OSM is open data) and the optional
   `osmnx`/`networkx` packages. If either is unavailable (no internet, or
   package not installed) it falls back to a same-dataset nearest-alternate-
   corridor suggestion, so the feature degrades gracefully instead of
   crashing the app.

4. geocode_address() / compute_deployment_points() - lets a user type a
   place name instead of hunting for coordinates (free OpenStreetMap
   Nominatim search, no API key), and turns a manpower/barricade COUNT into
   actual map positions around the incident so the app can show "where"
   alongside "how many". Positions are geometrically spaced, not GPS-exact
   real-world placements - see compute_deployment_points' docstring.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0
METERS_PER_DEGREE_LAT = 111_320.0


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_spatial_index(df: pd.DataFrame):
    """BallTree over historical events in radians, haversine metric. O(log n)
    nearest-neighbour queries instead of an O(n) scan on every dashboard
    interaction."""
    from sklearn.neighbors import BallTree

    coords_rad = np.radians(df[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords_rad, metric="haversine")
    return tree


def nearest_context(lat: float, lon: float, df: pd.DataFrame, tree=None, k: int = 5) -> dict:
    """Returns the most common corridor/police_station/zone among the k
    nearest historical events to (lat, lon) - i.e. 'what part of the city
    grid is this new point most like'."""
    if tree is None:
        tree = build_spatial_index(df)
    query = np.radians([[lat, lon]])
    dist, idx = tree.query(query, k=min(k, len(df)))
    nearby = df.iloc[idx[0]]
    dist_km = dist[0] * EARTH_RADIUS_KM

    def _mode(s):
        s = s.dropna()
        return s.mode().iloc[0] if not s.empty else None

    return {
        "corridor": _mode(nearby["corridor"]),
        "police_station": _mode(nearby["police_station"]),
        "zone": _mode(nearby["zone"]),
        "is_corridor": int(_mode(nearby["is_corridor"])) if "is_corridor" in nearby else None,
        "nearest_distance_km": float(dist_km.min()),
        "n_neighbors_used": int(len(nearby)),
    }


def similar_incidents(df: pd.DataFrame, event_cause: str, lat: float, lon: float,
                       tree=None, k: int = 8, max_radius_km: float = 5.0,
                       same_cause_only: bool = True) -> pd.DataFrame:
    """The post-event-learning retrieval: most similar historical incidents
    by (cause, proximity), with their REAL observed outcomes. Falls back to
    "any cause" within radius if too few same-cause matches exist nearby, so
    a rare cause at a given spot still returns useful neighbours."""
    candidates = df
    if same_cause_only:
        same_cause = df[df["event_cause"] == event_cause]
        candidates = same_cause if len(same_cause) >= 3 else df

    if tree is None or same_cause_only is False or len(candidates) != len(df):
        sub_tree = build_spatial_index(candidates)
    else:
        sub_tree = tree

    query = np.radians([[lat, lon]])
    k_eff = min(k, len(candidates))
    dist, idx = sub_tree.query(query, k=k_eff)
    out = candidates.iloc[idx[0]].copy()
    out["distance_km"] = dist[0] * EARTH_RADIUS_KM
    out = out[out["distance_km"] <= max_radius_km]
    cols = ["event_cause", "address", "corridor", "priority",
            "requires_road_closure", "duration_min", "duration_valid",
            "start_ts", "distance_km"]
    return out[[c for c in cols if c in out.columns]].sort_values("distance_km")


def nearby_distinct_corridors(df: pd.DataFrame, lat: float, lon: float, k: int = 3,
                               exclude_corridor: str | None = None) -> list[dict]:
    """Offline fallback diversion suggestion: nearest distinct named corridors
    to a point, ranked by distance, excluding the corridor that's closed."""
    named = df[df["corridor"] != "Non-corridor"][["corridor", "latitude", "longitude"]]
    centroids = named.groupby("corridor")[["latitude", "longitude"]].mean().reset_index()
    if exclude_corridor:
        centroids = centroids[centroids["corridor"] != exclude_corridor]
    centroids["distance_km"] = _haversine_km(lat, lon, centroids["latitude"], centroids["longitude"])
    top = centroids.sort_values("distance_km").head(k)
    return top.to_dict("records")


def get_diversion_suggestion(lat: float, lon: float, df: pd.DataFrame,
                              corridor: str | None = None) -> dict:
    """Tries live OSM-based routing first (free, open data, needs internet);
    falls back to the dataset-driven nearest-corridor suggestion if OSM
    tooling/network isn't available. Always returns a usable result."""
    try:
        import osmnx as ox
        import networkx as nx

        G = ox.graph_from_point((lat, lon), dist=1200, network_type="drive")
        center_node = ox.distance.nearest_nodes(G, lon, lat)
        neighbors = list(G.neighbors(center_node))
        if not neighbors:
            raise ValueError("isolated node, no alternate edges")

        # simulate the closure: drop the node, see where traffic would have
        # to be rerouted from each neighbouring approach
        G_closed = G.copy()
        G_closed.remove_node(center_node)
        alt_routes = []
        for n in neighbors[:3]:
            for m in neighbors[:3]:
                if n == m or not nx.has_path(G_closed, n, m):
                    continue
                path = nx.shortest_path(G_closed, n, m, weight="length")
                names = []
                for u, v in zip(path[:-1], path[1:]):
                    edge = G_closed.get_edge_data(u, v)[0]
                    nm = edge.get("name")
                    if isinstance(nm, list):
                        nm = nm[0]
                    if nm:
                        names.append(nm)
                if names:
                    alt_routes.append(list(dict.fromkeys(names))[:4])
        return {
            "mode": "live_osm",
            "approach_count": len(neighbors),
            "suggested_routes": alt_routes[:3],
        }
    except Exception:
        alts = nearby_distinct_corridors(df, lat, lon, k=3, exclude_corridor=corridor)
        return {
            "mode": "historical_fallback",
            "approach_count": None,
            "suggested_routes": [[a["corridor"]] for a in alts],
            "route_points": [{"corridor": a["corridor"], "lat": a["latitude"], "lon": a["longitude"]}
                              for a in alts],
            "note": "Live OSM road-graph routing unavailable in this run "
                    "(needs internet + osmnx) - showing nearest historical "
                    "corridors instead.",
        }


def offset_point(lat: float, lon: float, distance_m: float, bearing_deg: float) -> tuple[float, float]:
    """Move (lat, lon) by distance_m metres along compass bearing bearing_deg
    (0=N, 90=E). Flat-earth approximation - accurate to well under a metre
    of error at the ~50-300m scale this is used for (placing markers around
    an incident), nowhere near enough distance for earth curvature to matter."""
    bearing_rad = math.radians(bearing_deg)
    dlat = (distance_m * math.cos(bearing_rad)) / METERS_PER_DEGREE_LAT
    meters_per_deg_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(lat))
    dlon = (distance_m * math.sin(bearing_rad)) / meters_per_deg_lon
    return lat + dlat, lon + dlon


def bearing_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing (0-360, 0=N) from point 1 to point 2."""
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _deployment_points_geometric(lat, lon, n_barricades, n_personnel,
                                  barricade_radius_m=130, cluster_radius_m=28):
    """The always-available fallback: an evenly-spaced ring, no road-network
    awareness. See compute_deployment_points for why this exists and when
    it's used."""
    barricades = []
    for i in range(max(0, n_barricades)):
        bearing = (360.0 / n_barricades) * i
        barricades.append(offset_point(lat, lon, barricade_radius_m, bearing))

    officers = []
    at_barricade = min(max(0, n_personnel), len(barricades))
    officers.extend(barricades[:at_barricade])

    remaining = max(0, n_personnel - at_barricade)
    for i in range(remaining):
        bearing = (360.0 / remaining) * i if remaining > 1 else 0.0
        officers.append(offset_point(lat, lon, cluster_radius_m, bearing))

    return {"barricades": barricades, "officers": officers,
            "anchor": (lat, lon), "mode": "geometric_fallback", "n_approaches": None}


def _deployment_points_live(lat, lon, n_barricades, n_personnel,
                             barricade_dist_m=90, cluster_radius_m=24, search_radius_m=400):
    """Road-aware placement: snaps to the nearest real junction and points
    barricades along the bearings of its ACTUAL connecting roads, instead of
    arbitrary evenly-spaced compass directions. This is what makes a
    4-way crossroads look different from a T-junction or a roundabout,
    rather than every location producing the same ring pattern. Requires
    `osmnx` + internet; raises on any failure so the caller falls back."""
    import osmnx as ox

    G = ox.graph_from_point((lat, lon), dist=search_radius_m, network_type="drive")
    center_node = ox.distance.nearest_nodes(G, lon, lat)
    clat, clon = G.nodes[center_node]["y"], G.nodes[center_node]["x"]

    neighbors = list(dict.fromkeys(G.neighbors(center_node)))  # de-dup, keep order
    if not neighbors:
        raise ValueError("nearest node has no connecting roads")

    bearings = sorted(
        bearing_between(clat, clon, G.nodes[n]["y"], G.nodes[n]["x"]) for n in neighbors
    )

    n_b = min(max(0, n_barricades), len(bearings))
    barricades = [offset_point(clat, clon, barricade_dist_m, b) for b in bearings[:n_b]]

    officers = []
    at_barricade = min(max(0, n_personnel), len(barricades))
    officers.extend(barricades[:at_barricade])
    remaining = max(0, n_personnel - at_barricade)
    for i in range(remaining):
        bearing = (360.0 / remaining) * i if remaining > 1 else 0.0
        officers.append(offset_point(clat, clon, cluster_radius_m, bearing))

    return {"barricades": barricades, "officers": officers,
            "anchor": (clat, clon), "mode": "live_osm", "n_approaches": len(neighbors)}


def compute_deployment_points(lat: float, lon: float, n_barricades: int, n_personnel: int,
                               barricade_radius_m: float = 130, cluster_radius_m: float = 28,
                               try_live: bool = True) -> dict:
    """Turns a manpower COUNT and a barricade-point COUNT into actual map
    positions, since the models only output how many, not where exactly.

    Tries real road-network placement first (free, via OSMnx/OpenStreetMap -
    see _deployment_points_live): barricades go along the actual bearings of
    the nearest junction's real connecting roads, so different locations
    genuinely look different from each other. Falls back to an evenly-spaced
    geometric ring around the raw point when osmnx/internet aren't available
    - the previous behaviour, kept as a safety net, not the primary path.

    Either way, one officer is placed at each barricade (a barricade
    unstaffed is just a cone) and any remaining personnel cluster tightly
    near the incident as the on-scene team. Result includes "mode" so the
    caller can be transparent about which path produced it.
    """
    if try_live:
        try:
            return _deployment_points_live(lat, lon, n_barricades, n_personnel)
        except Exception:
            pass
    return _deployment_points_geometric(lat, lon, n_barricades, n_personnel,
                                         barricade_radius_m, cluster_radius_m)


def geocode_address(query: str, viewbox: tuple | None = None,
                     city_hint: str = "Bengaluru, Karnataka, India",
                     timeout: float = 6.0) -> dict | None:
    """Free, keyless geocoding via OpenStreetMap Nominatim - lets a user type
    'Silk Board Junction' instead of hunting for coordinates. Requires
    internet access at runtime (works wherever this app is actually run or
    deployed; this sandbox used during development has none, which is
    exactly the failure path the broad except below is built to handle
    gracefully rather than crash the form).

    Respects Nominatim's usage policy for the free public endpoint: a
    descriptive User-Agent, a single request per call (no batching), and a
    short timeout. For anything beyond light, occasional lookups like this,
    Nominatim's own docs ask you to self-host or use a paid provider -
    appropriate for a hackathon prototype, not for production traffic.

    Returns {"lat", "lon", "display_name"} or None if nothing was found or
    the request failed for any reason (no internet, rate-limited, etc).
    """
    if not query or not query.strip():
        return None
    try:
        import requests
    except ImportError:
        return None

    q = query.strip()
    if city_hint and city_hint.split(",")[0].lower() not in q.lower():
        q = f"{q}, {city_hint}"

    params = {"q": q, "format": "json", "limit": 1}
    if viewbox:
        params["viewbox"] = f"{viewbox[0]},{viewbox[1]},{viewbox[2]},{viewbox[3]}"

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers={"User-Agent": "astram-event-impact-forecaster/1.0 (hackathon prototype)"},
            timeout=timeout,
        )
        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None
        hit = results[0]
        return {
            "lat": float(hit["lat"]),
            "lon": float(hit["lon"]),
            "display_name": hit.get("display_name", query),
        }
    except Exception:
        return None


def coverage_ring_points(lat: float, lon: float, radius_m: float, n_points: int = 36) -> tuple[list, list]:
    """Generate lat/lon points forming a ring of radius_m metres around
    (lat, lon).  Used to draw a coverage-radius circle on the deployment map
    without needing any Mapbox shape-layer API - just a close-spaced
    Scattermapbox line trace."""
    lats, lons = [], []
    for i in range(n_points + 1):
        bearing = 360.0 * i / n_points
        plat, plon = offset_point(lat, lon, radius_m, bearing)
        lats.append(plat)
        lons.append(plon)
    return lats, lons
