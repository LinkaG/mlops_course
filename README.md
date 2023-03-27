# MLOps_course
Практические задания по mlops

## Простейший конвейер машинного обучения

### Цель задания:
В этом задании требуется создать автоматический конвейер проекта машинного обучения. Для склейки отдельных частей конвейера необходимо применить простые скрипты автоматизации.

### Описание решения:
Идея для реализации из соревнования [Data Fusion Contest 2023](https://ods.ai/competitions/data-fusion2023-attack) по теме Adversarial ML.

В нашем распоряжении имеется банковская модель классификации, предсказывающая дефолт клиента. Это рекуррентная нейросеть, принимающая на вход последние 300 транзакций клиента и классифицирующая клиентов на 2 класса. У нас нет доступа к полному набору данных, на которых модель была обучена, однако есть небольшая размеченная выборка клиентов с сопроводительными материалами.

Алгоритм по последовательности транзакций создает новый табличный .csv файл, который сильнее всего поменяет предсказания в предоставленной модели. 

### Запуск решения:

- установка зависимостей

```
pip3 install -r requirements
```
- запуск конвейера

```
./pipline.sh
```

### Этапы решения

- 1 этап: загружается архивы датасета с транзакциями и целевой переменной, распаковываются в папку data/external; также загружается модель для атаки;

- 2 этап: запускается скрипт для подготовки датасета для использования в RNN-модели;

- 3 этап: запускается скрипт для получения атакованных транзакций;

- 4 этап: запускается скрипт для проверки полученных транзакций.
