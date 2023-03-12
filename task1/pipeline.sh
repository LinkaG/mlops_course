#!/usr/bin/env bash

rm -rf ./data
mkdir ./data

mkdir ./data/external
mkdir ./data/interim
mkdir ./data/processed

rm -rf ./model
mkdir ./model



#загружаем архивы датасета с транзакциями и целевой переменной, распаковываем в папку data
wget -qO- https://storage.yandexcloud.net/ds-ods/files/materials/c7b69754/transactions.zip | zcat >> ./data/external/transactions.csv
wget -qO- https://storage.yandexcloud.net/ds-ods/files/materials/a4faa80b/train_target.zip | zcat >> ./data/external/train_target.csv

#загружаем модель RNN бинарного классификатора, которую собираемся атаковать
wget -qO- -O ./model/tmp.zip https://storage.yandexcloud.net/ds-ods/files/materials/750fd067/model.zip && unzip ./model/tmp.zip && rm -f ./model/tmp.zip
wget -O ./model/quantiles.json https://storage.yandexcloud.net/ds-ods/files/materials/4892996d/quantiles.json

#запускаем скрипт для подготовки датасета для использования в RNN-модели
PYTHONPATH=. python3 src/features/data_preprocessing.py

#запускаем скрипт для получения атакованных транзакций
PYTHONPATH=. python3 src/models/model_prepatation.py

#запускаем скрипт для проверки полученных транзакций
PYTHONPATH=. python -m unittest src/tests/model_testing.py




