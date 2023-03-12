import json
import pickle
import random
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.autograd.functional import jacobian
from tqdm import tqdm


from model import TransactionsDataset, TransactionsRnn, get_dataloader, process_for_nn


bins_path = 'model/nn_bins.pickle'
with open(bins_path, "rb") as f:
    bins = pickle.load(f)

features = bins.pop("features")
features_to_operate = ['mcc_code', 'transaction_amt']


def convert_budget_constraints (quantiles_path, bins):
    """
    Метод делает преобразование файла с исходными ограничениями
    по изменению транзакций в удобную для обработки форму.
    """
    with open(quantiles_path, "rb") as f:
        quantiles = json.load(f)

    # преобразовываем бюджетные ограничения так, чтобы потом было удобно
    negative_q_df = pd.DataFrame([
        list(quantiles['negative']['min'].keys())], index=['mcc_code']).T
    negative_q_df['minqua'] = [quantiles['negative']['min'][k] for k in negative_q_df.mcc_code]
    negative_q_df['maxqua'] = [quantiles['negative']['max'][k] for k in negative_q_df.mcc_code]

    positive_q_df = pd.DataFrame([
        list(quantiles['positive']['min'].keys())], index=['mcc_code']).T
    positive_q_df['minqua'] = [quantiles['positive']['min'][k] for k in positive_q_df.mcc_code]
    positive_q_df['maxqua'] = [quantiles['positive']['max'][k] for k in positive_q_df.mcc_code]

    temp_bins_mapper = {
        'minqua': 'transaction_amt',
        'maxqua': 'transaction_amt'
    }

    def digitize_cat(df):
        for dense_col in list(bins.keys()) + ['minqua', 'maxqua']:
            if dense_col in df:
                if dense_col in ["transaction_amt", 'minqua', 'maxqua']:
                    df[dense_col + '_bin'] = pd.cut(df[dense_col],
                                                    bins=bins[temp_bins_mapper.get(dense_col, dense_col)],
                                                    labels=False).astype(int)
                else:
                    df[dense_col + '_bin'] = pd.cut(
                        df[dense_col].astype(float).astype(int), bins=bins[temp_bins_mapper.get(dense_col, dense_col)],
                        labels=False,
                    ).astype(int)

        return df

    negative_q_df_after_digitize = negative_q_df.pipe(digitize_cat)
    negative_q_df_after_digitize = negative_q_df_after_digitize.set_index('mcc_code')

    positive_q_df_after_digitize = positive_q_df.pipe(digitize_cat)
    positive_q_df_after_digitize = positive_q_df_after_digitize.set_index('mcc_code')

    return quantiles, negative_q_df_after_digitize, positive_q_df_after_digitize


def found_source(original2bin, feature, bin_position):
    """восстановления айдишников эмбеддингов в номера транзакций."""
    for bin_key in original2bin[feature].keys():
        if original2bin[feature][bin_key] == bin_position:
            return bin_key
    return -1  # dirty hack


def _compute_best_option_for_change_emb(original_emb, desired, emb, feature_name):
    """Метод находит лучшую замену из имеющихся эмбеддингов по направлению, указанному якобианом."""
    desired = desired.reshape(desired.shape[0], desired.shape[1], 1, desired.shape[2])
    original_emb = original_emb.reshape(original_emb.shape[0], original_emb.shape[1], 1, original_emb.shape[2])
    embs2choose = emb.weight.repeat(original_emb.shape[0], original_emb.shape[1], 1, 1)

    similarity2possible_from_desired = -100500 * torch.ones(
        (
            original_emb.shape[0],
            original_emb.shape[1],
            emb.weight.shape[0]
        ),
        device=emb.weight.device
    )

    desired_direction = desired / torch.norm(desired, dim=-1, keepdim=True)
    possible_steps_direction = embs2choose - original_emb
    possible_steps_direction = possible_steps_direction / (
                torch.norm(possible_steps_direction, dim=-1, keepdim=True) + 0.1)

    for batch_item in range(original_emb.shape[0]):
        similarity2possible_from_desired[batch_item] = torch.bmm(
            desired[batch_item],
            possible_steps_direction[batch_item].transpose(2, 1)
        ).reshape(desired.shape[1], emb.num_embeddings)

    return similarity2possible_from_desired


def chunk_embeddings_by_type(features, features_to_operate, embs_list, original_emb, desired):
    """
    метод нарезает сконкатенированный вектор эмбеддингов обратно в эмбеддинги мцц-кодов,
    сумм транзакций и т.д.
    """
    desired_separated = []
    original_separated = []
    operated_features = []
    operated_embeddings = []

    prev_emb_pointer = 0
    for feature, emb in zip(features, embs_list):
        if feature not in features_to_operate:
            prev_emb_pointer = prev_emb_pointer + emb.embedding_dim
            continue
        operated_features.append(feature)
        operated_embeddings.append(emb)
        desired_separated.append(desired[:, :, prev_emb_pointer: prev_emb_pointer + emb.embedding_dim])
        original_separated.append(original_emb[:, :, prev_emb_pointer: prev_emb_pointer + emb.embedding_dim])
        prev_emb_pointer = prev_emb_pointer + emb.embedding_dim

    return desired_separated, original_separated, operated_features, operated_embeddings


def chose_transaction_embs(original_emb, desired, embs_list, features, features_to_operate, is_pos_amt, budget=10):
    """
    метод оценивает, какие замены эмбеддингов в паре транзакции-суммы
    даст наибольший вклад в "слом" модели
    """
    assert len(desired.shape) == 3, desired.shape
    assert desired.shape == original_emb.shape, (desired.shape, original_emb.shape)

    desired_separated, original_separated, operated_features, operated_embeddings = chunk_embeddings_by_type(
        features,
        features_to_operate,
        embs_list,
        original_emb,
        desired)

    similarity2possible_from_desired = {}

    for o, d, emb, f in zip(original_separated, desired_separated, operated_embeddings, operated_features):
        similarity2possible_from_desired[f] = _compute_best_option_for_change_emb(o, d, emb, f)

    mcc_impact_over_transactions = similarity2possible_from_desired['mcc_code']
    amt_impact_over_transactions = similarity2possible_from_desired['transaction_amt']

    impact_scores_overall = mcc_impact_over_transactions.reshape(*mcc_impact_over_transactions.shape, 1) \
                            + amt_impact_over_transactions.reshape(*amt_impact_over_transactions.shape[:-1], 1,
                                                                   amt_impact_over_transactions.shape[-1])

    assert mcc_impact_over_transactions.shape[0] == amt_impact_over_transactions.shape[0]
    assert mcc_impact_over_transactions.shape[1] == amt_impact_over_transactions.shape[1]

    assert impact_scores_overall.shape == (
        mcc_impact_over_transactions.shape[0],
        mcc_impact_over_transactions.shape[1],
        mcc_impact_over_transactions.shape[2],
        amt_impact_over_transactions.shape[2]
    ), impact_scores_overall.shape

    best_overall_impacts = impact_scores_overall

    best_amts_ids = impact_scores_overall.argmax(dim=-1)
    best_overall_impacts = best_overall_impacts.max(dim=-1).values
    best_mccs_ids = best_overall_impacts.argmax(dim=-1)
    best_overall_impacts_temp = best_overall_impacts
    best_overall_impacts = best_overall_impacts.max(dim=-1).values
    best_transactions = best_overall_impacts.argsort(descending=True, dim=-1)[:, :budget]

    def choose1(index, target):
        result = torch.zeros(*index.shape, dtype=target.dtype, device=target.device)
        for i in range(index.shape[0]):
            result[i] = target[i][index[i]]
        return result

    def choose2(index, target):
        result = torch.zeros(*index.shape, dtype=target.dtype, device=target.device)
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                result[i, j] = target[i, j][index[i, j]]
        return result

    def choose3(index, target):
        result = torch.zeros(*target.shape, dtype=target.dtype, device=target.device)
        for i in range(index.shape[0]):
            for j in range(index.shape[1]):
                result[i, j] = target[i, index[i, j]]
        return result

    chosen_transactions = best_transactions
    chosen_mccs_ids = choose1(chosen_transactions, best_mccs_ids)
    chosen_amts_ids = choose2(chosen_mccs_ids, choose3(chosen_transactions, best_amts_ids))

    return operated_features, ((chosen_transactions), [chosen_mccs_ids, chosen_amts_ids])


def estimate_best_exchange(rnn, record, features, features_to_operate, is_pos_amt, total_changes=10, to_positive=True):
    x = rnn._get_input_embs(record)

    assert len(x.shape) == 3, x.shape

    jac = jacobian(lambda x: rnn.classify_emb(x.shape[0], *rnn.get_emb(x)), x)

    operated_features, impactor_ids = chose_transaction_embs(
        x,
        jac[
            torch.arange(jac.shape[0]).to(jac.device),
            to_positive,
            torch.arange(jac.shape[0]).to(jac.device)],
        rnn._transaction_cat_embeddings,
        features,
        features_to_operate,
        is_pos_amt,
        budget=total_changes
    )

    return operated_features, impactor_ids


def uncut(original_df, original_transaction_id, transaction_relative_ids, transaction_details_ids, bins,
          features_to_operate):
    """
    метод находит для индексов эмбеддингов исходные суммы и названия мцц кодов транзакций для
    каждой записи
    """
    transactions_to_change_ids = original_transaction_id[transaction_relative_ids]
    for transaction_id, change_ids in zip(transactions_to_change_ids, range(transactions_to_change_ids.shape[0])):
        changes = dict(zip(features_to_operate, transaction_details_ids))

        mcc_code_original = found_source(original2bin, 'mcc_code', changes['mcc_code'][change_ids].item())
        amt_change_bucket = changes['transaction_amt'][change_ids].item()

        amt_change = None
        if mcc_code_original in positive_q_df_after_digitize.index:
            mcc_qua_record = positive_q_df_after_digitize.loc[mcc_code_original]

            if mcc_qua_record.minqua_bin <= amt_change_bucket <= mcc_qua_record.maxqua_bin:
                bucket_amt_left_bound = bins[feature][changes['transaction_amt'][change_ids].item()]
                bucket_amt_right_bound = bins[feature][changes['transaction_amt'][change_ids].item() + 1]
                amt_change = (max(mcc_qua_record.minqua, bucket_amt_left_bound) + min(mcc_qua_record.maxqua,
                                                                                      bucket_amt_right_bound)) / 2

        if mcc_code_original in negative_q_df_after_digitize.index:
            mcc_qua_record = negative_q_df_after_digitize.loc[mcc_code_original]

            if mcc_qua_record.minqua_bin <= amt_change_bucket <= mcc_qua_record.maxqua_bin:
                bucket_amt_left_bound = bins[feature][changes['transaction_amt'][change_ids].item()]
                bucket_amt_right_bound = bins[feature][changes['transaction_amt'][change_ids].item() + 1]
                amt_change = (max(mcc_qua_record.minqua, bucket_amt_left_bound) + min(mcc_qua_record.maxqua,
                                                                                      bucket_amt_right_bound)) / 2

        if amt_change is None:
            amt_change = -1  # dirty hack
            mcc_code_original = -1
        #             raise Exception(f'mc {mcc_code_original}, amt {amt_change_bucket}')

        original_df[transaction_id.item(), features.index('transaction_amt')] = amt_change
        original_df[transaction_id.item(), features.index('mcc_code')] = mcc_code_original


def heal_budget(original_transactions, harmed_transactions, quantiles):
    """метод лечит итоговую атаку от срабатывания бюджетных ограничений."""
    assert original_transactions.shape == harmed_transactions.shape, 'nothing to do with the corrupted data'

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

    diff_count = defaultdict(int)
    need_restore_to_original_records_ids = []

    errors_found = defaultdict(int)

    for i, (a, b) in enumerate(
            tqdm(zip(original_transactions.itertuples(), harmed_transactions.itertuples()), desc='looking for errors',
                 total=original_transactions.shape[0])):
        if is_different_records(a, b):
            diff_count[a.user_id] += 1
            if diff_count[a.user_id] > BUDGET:
                errors_found['budget over'] += 1
                need_restore_to_original_records_ids.append(i)
                continue

            if np.sign(a.transaction_amt) != np.sign(b.transaction_amt):
                errors_found['sign change'] += 1
                need_restore_to_original_records_ids.append(i)
                continue

            if a.transaction_amt < 0:
                ruler = quantiles["negative"]
            else:
                ruler = quantiles["positive"]

            key_b = str(b.mcc_code)

            if key_b not in ruler["max"] or key_b not in ruler["min"]:
                errors_found['bad mcc'] += 1
                need_restore_to_original_records_ids.append(i)
                continue
            upper_bound_b = ruler["max"][key_b]
            lower_bound_b = ruler["min"][key_b]
            if any(
                    [
                        upper_bound_b < b.transaction_amt,
                        lower_bound_b > b.transaction_amt,
                    ]
            ):
                errors_found['amt exceeded'] += 1
                need_restore_to_original_records_ids.append(i)
                continue

    result = harmed_transactions.copy()

    for id2restore in tqdm(need_restore_to_original_records_ids, desc='healing errors'):
        result.iloc[id2restore] = original_transactions.iloc[id2restore]

    print(errors_found)

    return result


if __name__ == "__main__":
    bins_path = 'model/nn_bins.pickle'
    with open(bins_path, "rb") as f:
        bins = pickle.load(f)
    features = bins.pop("features")
    features_to_operate = ['mcc_code', 'transaction_amt']


    model_path = "model/nn_weights.ckpt"  # путь до файла с весами нейронной сети (nn_weights.ckpt)
    quantiles_path = "model/quantiles.json"  # путь до файла с квантилями для таргета (quantiles.pickle)
    BUDGET = 10  # разрешенное количество изменений транзакций для каждого пользователя
    output_path = "data/processed/non_naive_submission.csv"  # куда сохранить атакованные транзакции
    transactions_path = "data/external/transactions.csv"  # путь до файла с транзакциями, которые атакуются
    transactions_for_rnn_path = "data/interim/transactions_for_rnn.csv"

    print('Загружаем модель для атаки...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rnn = TransactionsRnn()
    rnn.load_state_dict(torch.load(model_path))
    pl.seed_everything(20230206)
    rnn = rnn.eval()
    rnn._gru.train()
    rnn = rnn.to(device)

    # преобразовываем бюджетные ограничения
    quantiles, negative_q_df_after_digitize, positive_q_df_after_digitize = convert_budget_constraints(quantiles_path, bins)

    # для восстановления айдишников эмбеддингов в номера транзакций
    original2bin = {
        'mcc_code': {
            mcc:(negative_q_df_after_digitize.mcc_code_bin.get(mcc) or positive_q_df_after_digitize.mcc_code_bin.get(mcc))
            for mcc in set(list(negative_q_df_after_digitize.index) + list(positive_q_df_after_digitize.index))
        }
    }

    #загружаем исходный датасет
    df_transactions = (
        pd.read_csv(
            transactions_path,
            parse_dates=["transaction_dttm"],
            dtype={"user_id": int, "mcc_code": int, "currency_rk": int, "transaction_amt": float}  # , nrows=300*400
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

    #загружаем датасет с обработанными транзакциями, завершаем подготовку данных для модели
    df_trans_for_rnn = pd.read_csv(transactions_for_rnn_path)
    df = process_for_nn(df_transactions, features, bins)
    dataset = TransactionsDataset(df)
    dataloader = get_dataloader(dataset, device, batch_size=10, is_validation=True)

    df_transactions_np = df_transactions.values

    tqdm_dl = tqdm(dataloader)


    print('Начинаем расчет транзакций для атаки модели')

    for xs, users, original_transaction_ids, is_pos_amt in tqdm_dl:
        score_at_start = rnn(xs)[:, 1]
        to_positive_mask = (score_at_start < 0.15).bool()

        for _ in range(1):
            operated_features, (
            batched_transaction_relative_ids, batched_transaction_details_ids) = estimate_best_exchange(
                rnn,
                xs,
                features,
                features_to_operate,
                is_pos_amt.bool(),
                to_positive=to_positive_mask.long(),
                total_changes=10
            )

            for j in range(xs.shape[0]):
                for i, transaction2change in enumerate(batched_transaction_relative_ids[j]):
                    for feature, change_modality in zip(operated_features,
                                                        [x[j] for x in batched_transaction_details_ids]):
                        xs[j, features.index(feature), transaction2change] = change_modality[i]

            for original_transaction_id, transaction_relative_ids, transaction_details_batch_index in zip(
                    original_transaction_ids, batched_transaction_relative_ids,
                    range(original_transaction_ids.shape[0])):
                uncut(
                    df_transactions_np,
                    original_transaction_id.cpu(),
                    transaction_relative_ids.cpu(),
                    [x[transaction_details_batch_index].cpu() for x in batched_transaction_details_ids],
                    bins,
                    operated_features)

    for i, column in enumerate(df_transactions.columns):
        df_transactions[column] = df_transactions_np[:, i]

    print('Чиним атакованные транзакции')
    harmed_transactions = df_transactions.drop(['hour', 'day', 'month', 'number_day', 'is_pos_amt'], axis=1)
    original_transactions = pd.read_csv(
        transactions_path,
        parse_dates=["transaction_dttm"],
        dtype={"user_id": int, "mcc_code": int, "currency_rk": int, "transaction_amt": float}  # , nrows=300*400
    )
    healed_dangerous_transactions = heal_budget(original_transactions, harmed_transactions, quantiles)
    healed_dangerous_transactions.to_csv(output_path, index=False)












