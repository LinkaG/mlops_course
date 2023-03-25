import pandas as pd
import click
import geopandas as gpd
import requests
from shapely.geometry import Point


@click.command()
@click.argument("query_string", type=click.STRING)
@click.argument("output_path", type=click.Path())
def get_osm_cafes_data(query_string: str, output_path: str):
    """Make geodataframe from Overpass API
    :param query_string: Ovepass API query string
    :param output_path: Path to save cafe data
    :return:
    """
    # Retrieve URL contents
    r = requests.get(query_string)

    # Make dataframe
    df = pd.DataFrame(r.json()['elements'])

    # Convert to geodataframe
    df['geometry'] = [Point(xy) for xy in zip(df.lon, df.lat)]
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    gdf.to_file(output_path, driver='GeoJSON')


if __name__ == "__main__":
    get_osm_cafes_data()

