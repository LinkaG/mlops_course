#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import unittest


class TestModel(unittest.TestCase):

    SOURCE_PATH: str = "data/external/transactions.csv"
    ATTACK_PATH: str = "data/processed/non_naive_submission.csv"
    QUANTILE_PATH: str = "model/quantiles.json"

    BUDGET: int = 10

    def setUp(self) -> None:
        with open(self.QUANTILE_PATH, "rb") as f:
            self.quantiles = json.load(f)

        self.df_source = pd.read_csv(
            self.SOURCE_PATH,
            parse_dates=["transaction_dttm"],
            dtype={"user_id": int, "mcc_code": int, "currency_rk": int, "transaction_amt": float},
        ).sort_values(by=["user_id", "transaction_dttm"])

        self.df_attack = pd.read_csv(
            self.ATTACK_PATH,
            parse_dates=["transaction_dttm"],
            dtype={"user_id": int, "mcc_code": int, "currency_rk": int, "transaction_amt": float},
        ).sort_values(by=["user_id", "transaction_dttm"])

        if self.df_source.shape != self.df_attack.shape:
            self.fail("Some error message")

    def tearDown(self) -> None:
        pass

    @staticmethod
    def is_different_records(a, b):
        return not all(
            [
                a.user_id == b.user_id,
                a.mcc_code == b.mcc_code,
                a.currency_rk == b.currency_rk,
                np.isclose(a.transaction_amt, b.transaction_amt),
                a.transaction_dttm == b.transaction_dttm,
            ]
        )

    def test_correct_budget(self) -> None:
        diff_count = defaultdict(int)
        print("Start testing")
        for a, b in tqdm(
            zip(self.df_source.itertuples(), self.df_attack.itertuples()),
            total=self.df_source.shape[0]
        ):
            if self.is_different_records(a, b):
                diff_count[a.user_id] += 1
                self.assertGreaterEqual(self.BUDGET, diff_count[a.user_id], msg="budget over")

                self.assertEqual(
                    np.sign(a.transaction_amt), np.sign(b.transaction_amt),
                    msg="sign changed"
                )

                if a.transaction_amt < 0:
                    ruler = self.quantiles["negative"]
                else:
                    ruler = self.quantiles["positive"]

                key_b = str(b.mcc_code)
                upper_bound_b = ruler["max"][key_b]
                lower_bound_b = ruler["min"][key_b]

                self.assertFalse(
                    any([upper_bound_b < b.transaction_amt, lower_bound_b > b.transaction_amt]),
                    msg="amt exceeded"
                )
