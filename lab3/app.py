import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

# Загружаем обученную модель
model = joblib.load("./model.joblib")


class IrisSpecies(BaseModel):
    """
    Схема входных данных
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post('/predict')  # описываем постзапрос
def predict(iris: IrisSpecies):
    """
    Эндпоинт принимает фичи и делает предсказания
    """
    features = [list(iris.dict().values())]
    prediction = model.predict(features).tolist()[0]
    return {"prediction": prediction}


if __name__=='__main__':
    # запускаем сервер
    uvicorn.run(app, host='127.0.0.1', port=5000)
