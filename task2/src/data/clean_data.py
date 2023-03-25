import pandas as pd
import click
import yaml

MIN_AREA = 15  # Outlier range for floor area
MAX_AREA = 300

MIN_PRICE = 1_000_000  # Outlier range for price
MAX_PRICE = 100_000_000

MIN_KITCHEN = 3  # Outlier range for kitchen area
MAX_KITCHEN = 70


@click.command()
@click.argument("input_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def clean_data(input_path: str, output_path: str):
    """Function removes excess columns and enforces
    correct data types.
    :param input_path: Path to read filtered by region DataFrame
    :param output_path: Path to save cleaned DataFrame
    :return:
    """
    print('Чистим данные')
    df = pd.read_csv(input_path)
    df.drop("time", axis=1, inplace=True)
    # Column actually contains -1 and -2 values presumably for studio apartments.
    df["rooms"] = df["rooms"].apply(lambda x: 0 if x < 0 else x)
    df["price"] = df["price"].abs()  # Fix negative values
    # Drop price and area outliers.
    df = df[(df["area"] <= MAX_AREA) & (df["area"] >= MIN_AREA)]
    df = df[(df["price"] <= MAX_PRICE) & (df["price"] >= MIN_PRICE)]
    # Fix kitchen area outliers.
    # At first, replace all outliers with 0.
    df.loc[
        (df["kitchen_area"] >= MAX_KITCHEN) | (df["kitchen_area"] <= MIN_KITCHEN),
        "kitchen_area",
    ] = 0
    # Then calculate kitchen area based on the floor area, except for studios.
    area_mean, kitchen_mean = df[["area", "kitchen_area"]].quantile(0.5)
    kitchen_share = kitchen_mean / area_mean
    df.loc[(df["kitchen_area"] == 0) & (df["rooms"] != 0), "kitchen_area"] = (
        df.loc[(df["kitchen_area"] == 0) & (df["rooms"] != 0), "area"] * kitchen_share
    )

    df.to_csv(output_path, index=False)



if __name__ == "__main__":
    clean_data()

