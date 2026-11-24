"""
GPS life space computation. Features include:
- Convex hull area computation
- Convex hull vertex extraction

@author Henry Renninger
"""


import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_008.8

def compute_geographic_centroid(lats, lons):
    """
    Due to the curvature of the earth, centroid must be calcultated after projecting
    lat/lon to 3D cartesian coordinates. This function first map coordinates, and then
    compute the centroid in cartesian space, and maps it back to lat/lon space.

    Parameters
    ----------
    lats
        Flat array of latitudes (float)
    lons
        Flat array of longitudes (float)
    
    Returns
    ----------
    lat_0
        Latitude of the centroid (float)
    lon_0
        Longitude of the centroid (float)
    """
    # Convert lat/lon to radians
    lats_rad = np.deg2rad(lats)
    lons_rad = np.deg2rad(lons)

    # Get x/y/z 3D coordinates
    x = EARTH_RADIUS_M*np.cos(lats_rad)*np.cos(lons_rad)
    y = EARTH_RADIUS_M*np.cos(lats_rad)*np.sin(lons_rad)
    z = EARTH_RADIUS_M*np.sin(lats_rad)

    # Mean vectors
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    # Convert back to lat/lon
    lon_mean = np.arctan2(y_mean, x_mean)
    hyp = np.hypot(x_mean, y_mean)
    lat_mean = np.arctan2(z_mean, hyp)

    return float(np.rad2deg(lat_mean)), float(np.rad2deg(lon_mean))

def cartesian_transform(lats, lons, lat_0, lon_0):
    """
    Transforms lat/lon coordinates to x/y 2D cartesian coordinates
    for local geospatial computation and analysis.

    Parameters
    ----------
    lats
        Flat array of latitudes (float)
    lons
        Flat array of longitudes (float)
    lat_0
        Latitude of the centroid (float)
    lon_0
        Longitude of the centroid (float)

    Returns
    ----------
    x
        Flat array of x coordinates (float)
    y
        Flat array of y coordinates (float)    
    """
    # NEXT FEW LINES FIND A REFERENCE

    # Compute the shortest distance between the longitudes and the centroid longitude
    # Around the atimeridian, accounting for parallel shrinkage across latitudes
    d_lons = (lons - lon_0 + 180.0) % 360.0 - 180.0

    # Compute the shortest distance between the latitudes
    d_lats = lats - lat_0

    # Convert to radians
    d_lons_rad = np.deg2rad(d_lons)
    d_lats_rad = np.deg2rad(d_lats)
    lat_0_rad = np.deg2rad(lat_0)

    # Compute x/y coordinates in meters
    x = d_lons_rad* EARTH_RADIUS_M*np.cos(lat_0_rad)
    y = d_lats_rad*EARTH_RADIUS_M

    return x, y

def reverse_cartesian_transform(x, y, lat_0, lon_0):
    """
    Transforms x/y 2D cartesian coordinates back to lat/lon coordinates

    Parameters
    ----------
    x
        Flat array of x coordinates (float)
    y
        Flat array of y coordinates (float)
    lat_0
        Latitude of the centroid (float)
    lon_0
        Longitude of the centroid (float)

    Returns
    ----------
    lats
        Flat array of latitudes (float)
    lons
        Flat array of longitudes (float)
    """
    lat = lat_0 + np.rad2deg(y/EARTH_RADIUS_M)
    lon = lon_0 + np.rad2deg(x/(EARTH_RADIUS_M*np.cos(np.deg2rad(lat_0))))

    return lat, lon
    
def convex_hull(x, y):
    """
    Also known as Andrew's Monotone Chain Convex Hull Algoithm. Takes the mapped x/y
    coordinates and computes the convex hull of the points, returning the vertices of the
    the polygon in a counterclockwise order. Broken into an Upper and a Lower hull, the
    algorithm iteratively adds points to the hull, checking for collinearity and removing
    extraneous points. The final hull concatenates the Upper and Lower hulls.

    Parameters
    ----------
    x
        Flat array of x coordinates (float)
    y
        Flat array of y coordinates (float)

    Returns
    ----------
    hull
        2D array of the vertices of the convex hull in counterclockwise order (float)

    """
    # Use a sorted column stack to keep pairity
    points = list(np.unique(np.column_stack((x, y)), axis=0))
    
    # Cross product to determine the orientation of the points
    # i.e. whether they are collinear (0), clockwise (negative), or counterclockwise (positive)
    def cross(o, a, b):
        return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])
    
    # Lower hull construction (REVIEW)
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Upper hull construction (REVIEW)
    upper = []
    for p in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    # Remove the last point of each half because it is repeated at the beginning of the other half
    return np.array(lower[:-1] + upper[:-1])

def shoelace_area(hull):
    """
    Computes the area of a polygon the counterclockwise ordered vertices (shoelace algo)

    Parameters
    ----------
    hull
        2D array of convex hull vertices in counterclockwise order (float)

    Returns
    ----------
    area
        Area of the polygon (float)
    """
    x = hull[:, 0]
    y = hull[:, 1]

    S1 = np.dot(x, np.roll(y, -1))
    S2 = np.dot(y, np.roll(x, -1))

    area = .5*np.abs(S1 - S2)
    return area

def build_life_space_polygon(places):
    """
    Computes life space polygon and resprective area.

    Parameters
    ----------
    places
        DataFrame of places with representative lat/lon coordinates (float)

    Returns
    ----------
    area
        Area of the polygon (float)
    hull
        2D array of convex polygon vertices (hull) in counterclockwise order (float)
    """
    # Get latitudes and longitudes of the representative points of the places
    lats = pd.to_numeric(places["representative_latitude"], errors="coerce")
    lons = pd.to_numeric(places["representative_longitude"], errors="coerce")

    # NEED TO ADD A THRESHOLD FOR MINIMUM NUMBER OF PLACES

    # Compute centorid of the places
    lat_0, lon_0 = compute_geographic_centroid(lats, lons)

    # Transform lat/lon to x/y cartesian coordinates
    x, y = cartesian_transform(lats, lons, lat_0, lon_0)

    # Compute the convex hull of the points
    hull = convex_hull(x, y)

    # CHECK IF HULL IS EMPTY OR HAS LESS THAN 3 POINTS, RETURN EMPTY POLYGON

    # Compute the area of the convex hull
    area = shoelace_area(hull)

    # Reverse the hull x/y points to lat/lon coordinates
    hull_lats, hull_lons = reverse_cartesian_transform(hull[:, 0], hull[:, 1], lat_0, lon_0)

    return area, np.column_stack((hull_lats, hull_lons))
    