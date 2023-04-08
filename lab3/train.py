import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


if __name__=='__main__':
    # обучаем модель
    iris_data = load_iris()
    model = OneVsRestClassifier(LogisticRegression())
    model.fit(iris_data.data, iris_data.target)
    # записываем модель в рабочую директорию
    joblib.dump(model, "./model.joblib")
