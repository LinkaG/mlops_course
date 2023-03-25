import click
import pandas as pd
import joblib as jb
import lightgbm as lgb

from typing import List
from sklearn.metrics import mean_absolute_error, mean_squared_error



FEATURES = ['price', 'geo_lat', 'geo_lon', 'building_type', 'level', 'levels',
            'area', 'kitchen_area', 'object_type', 'year', 'month',
            'level_to_levels', 'area_to_rooms', 'cafes_0.012', 'cafes_0.08']


@click.command()
@click.argument("input_paths", type=click.Path(exists=True), nargs=2)
@click.argument("output_path", type=click.Path())
def train(input_paths: List[str], output_path: str):
    train_df = pd.read_csv(input_paths[0])
    test_df = pd.read_csv(input_paths[1])

    x_train = train_df.drop('price', axis=1)
    y_train = train_df['price']
    x_holdout = test_df.drop('price', axis=1)
    y_holdout = test_df['price']

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_holdout, y_holdout, reference=lgb_train)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l1'},
        'max_depth': 11,
        'num_leaves': 150,
        'learning_rate': 0.25,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'n_estimators': 1000,
        'bagging_freq': 2,
        'verbose': -1
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=200,
                    valid_sets=lgb_eval,
                    verbose_eval=False,
                    early_stopping_rounds=30)  # categorical_feature=['building_type']
    jb.dump(gbm, output_path)


if __name__ == "__main__":
    train()

