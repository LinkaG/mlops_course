import click
import pandas as pd
from typing import List


FEATURES = ['price', 'geo_lat', 'geo_lon', 'building_type', 'level', 'levels',
            'area', 'kitchen_area', 'object_type', 'year', 'month',
            'level_to_levels', 'area_to_rooms', 'cafes_0.012', 'cafes_0.08']


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_file_paths", type=click.Path(), nargs=2)
def prepare_datasets(input_filepath, output_file_paths: List[str]):
    df = pd.read_csv(input_filepath)
    df = df[FEATURES]
    df = df.drop_duplicates(subset=['geo_lat', 'geo_lon', 'level', 'area'], keep="last")

    train = df.sample(frac=0.75, random_state=200)
    test = df.drop(train.index)

    train.to_csv(output_file_paths[0], index=False)
    test.to_csv(output_file_paths[1], index=False)


if __name__ == "__main__":
    prepare_datasets()

