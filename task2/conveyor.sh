#!/usr/bin/env bash

#загружаем архив c датасетом и распаковываем его
wget -qO- https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/Jk1VPo4lpljgvA | zcat >> ./data/raw/all_v2.csv

#запускаем скрипт для выбора данных по определенному региону
PYTHONPATH=. python3 src/data/select_region.py data/raw/all_v2.csv data/interim/data_regional.csv 2661

#запускаем скрипт для очистки данных
PYTHONPATH=. python3 src/data/clean_data.py data/interim/data_regional.csv data/interim/data_cleaned.csv

# запускаем скрипт для добавления признаков
PYTHONPATH=. python3 src/features/add_features.py data/interim/data_cleaned.csv data/interim/data_featured.csv

# получаем osm для cafes
PYTHONPATH=. python3  src/data/get_osm_cafes_data.py "https://maps.mail.ru/osm/tools/overpass/api/interpreter?data=[out:json];nwr['addr:street'='Лиговский проспект']['addr:housenumber'=101];node[amenity=cafe](around:25000);out geom;" data/external/data_cafes.geojson

# получаем кафе в радиусе
PYTHONPATH=. python3 src/features/add_cafe_radius_features.py data/interim/data_featured.csv data/external/data_cafes.geojson data/processed/dataset.csv

# готовим датасет
PYTHONPATH=. python3 src/models/prepare_datasets.py data/processed/dataset.csv data/processed/train.csv data/processed/test.csv

# тренируем модель
PYTHONPATH=. python3 src/models/train.py data/processed/train.csv data/processed/test.csv models/model.clf

# оцениваем модель
PYTHONPATH=. python3 src/models/evaluate.py data/processed/test.csv models/model.clf reports/scores.csv
