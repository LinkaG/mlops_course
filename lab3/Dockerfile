FROM python:3.9.15-slim-buster as builder

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py /app/
WORKDIR /app/

FROM builder as trainer
WORKDIR /app/
RUN python train.py

FROM builder as api
COPY --from=trainer /app/model.joblib /app/model.joblib
WORKDIR /app/

EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]