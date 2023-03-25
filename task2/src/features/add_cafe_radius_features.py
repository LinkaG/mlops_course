import pandas as pd
import click
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

RADIUS_LIST = [0.002, 0.005, 0.08, 0.012]


def add_cafes_in_radius(gdf: gpd.GeoDataFrame, cafes_gdf: gpd.GeoDataFrame, radius: float):
    """Makes features based on number of cafes in certain radius
    :param gdf:
    :param cafes_gdf:
    :param radius:
    :return gdf:
    """
    gdf['cafes_' + str(radius)] = 0
    list_arrays = [ np.array((geom.xy[0][0], geom.xy[1][0])) for geom in cafes_gdf["geometry"] ]
    list_tuples = [tuple(x) for x in list_arrays]
    points_array = np.array(list_tuples)
    max_distance = radius
    for idx, geometry in gdf['geometry'].iteritems():
        current_flat_position = np.array([geometry.x, geometry.y])
        distance_array = np.sqrt(np.sum((points_array - current_flat_position) ** 2, 1))
        near_points = points_array[distance_array < max_distance]
        cafe_number = len(near_points)
        gdf['cafes_' + str(radius)].at[idx] = cafe_number
    return gdf


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("input_cafes_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def add_cafe_radius_features(input_path: str, input_cafes_path: str, output_path: str):
    """Makes features based on number of cafes in certain radius
    :param input_path:
    :param input_cafes_path:
    :param output_path:
    :return:
    """
    df = pd.read_csv(input_path)
    df['geometry'] = [Point(xy) for xy in zip(df.geo_lon, df.geo_lat)]
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    cafes_gdf = gpd.read_file(input_cafes_path)
    cafes_gdf = cafes_gdf.set_geometry('geometry')

    for radius in RADIUS_LIST:
        gdf = add_cafes_in_radius(gdf, cafes_gdf, radius)

    gdf.to_csv(output_path, index=False)


if __name__ == "__main__":
    add_cafe_radius_features()

