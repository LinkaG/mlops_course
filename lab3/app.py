import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Загружаем обученную модель
app = FastAPI()
model = joblib.load("./model.joblib")

class IrisSpecies(BaseModel):
    """
    класс для валидации входных данных.
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post('/predict') # описываем постзапрос
def predict(iris: IrisSpecies):
    """
    на вход принимаем фичи и делаем предсказания.
    """
    features = [[
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width
    ]]
    prediction = model.predict(features).tolist()[0]
    return {"prediction": prediction}


if __name__=='__main__':
    # запускаем сервер
    uvicorn.run(app, host='127.0.0.1', port = 80)