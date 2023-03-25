import pandas as pd
import click


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.argument("region", type=click.INT)
def select_region(input_path: str, output_path: str, region: int):
    """Function selects the listings belonging to a specified region.
    :param input_path: Path to read original DataFrame with all listings
    :param output_path: Path to save filtered DataFrame
    :param region: Selected region id
    :return:
    """
    print(f'выбираем данные, относящиеся к региону: {region}')
    df = pd.read_csv(input_path, nrows=100000)

    df = df[df['region'] == region]
    df.drop('region', axis=1, inplace=True)
    print(f'Selected {len(df)} samples in region {region}.')

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    select_region()

