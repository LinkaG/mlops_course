import pandas as pd
import pickle
import numpy as np


def process_for_nn(transactions_frame, features, nn_bins, *, need_padding=True):
    """Returns the dataset obtained from the source to feed the RNN model to the input.

    :param transactions_frame: source dataset with user transactions.
    :param features: categories of features.
    :param nn_bins: splitting features by bins.
    :param need_padding: optional parameter.
    :return: dataset prepared for the model.
    """
    transactions_frame = transactions_frame[transactions_frame["mcc_code"].notna()]

    assert set(nn_bins.keys()) <= set(transactions_frame.columns), "Something wrong"

    def digitize_cat(df):
        for dense_col in nn_bins.keys():
            if dense_col == "transaction_amt":
                df[dense_col] = pd.cut(df[dense_col], bins=nn_bins[dense_col], labels=False).astype(int)
            else:
                df[dense_col] = pd.cut(
                    df[dense_col].astype(float).astype(int), bins=nn_bins[dense_col], labels=False,
                )

        return df

    after_digitize = transactions_frame.pipe(digitize_cat)
    after_digitize = after_digitize.dropna()
    for dense_col in nn_bins.keys():
        after_digitize[dense_col] = after_digitize[dense_col].astype(int)


    num_transactions = 300
    if not need_padding:
        num_transactions = 1
    return (
        after_digitize.reset_index().groupby(["user_id"])[features + ['is_pos_amt', 'index']]
        # take last 300 transactions
        .apply(lambda x: x.values.transpose()[:, -num_transactions:].tolist())
        # additional padding to 300
        .apply(lambda x: np.array([list(i) + [0] * int(num_transactions - len(x[0])) for i in x]))
        .reset_index()
        .rename(columns={0: "sequences"})
    )


if __name__ == "__main__":
    print('Предобработка исходного датасета...')
    bins_path = "./model/nn_bins.pickle"
    source_file = "./data/external/transactions.csv"
    output_path = "./data/interim/transactions_for_rnn.csv"

    with open(bins_path, "rb") as f:
        bins = pickle.load(f)
    features = bins.pop("features")

    df_transactions = (
        pd.read_csv(
            source_file,
            parse_dates=["transaction_dttm"],
            dtype={"user_id": int, "mcc_code": int, "currency_rk": int, "transaction_amt": float},
        )
        .dropna()
        .assign(
            hour=lambda x: x.transaction_dttm.dt.hour,
            day=lambda x: x.transaction_dttm.dt.dayofweek,
            month=lambda x: x.transaction_dttm.dt.month,
            number_day=lambda x: x.transaction_dttm.dt.day,
            is_pos_amt=lambda x: x.transaction_amt > 0
        )
    )
    data_for_rnn = process_for_nn(df_transactions, features, bins)
    data_for_rnn.to_csv(output_path)
    print('Предобработанный датасет успешно сохранен!')
