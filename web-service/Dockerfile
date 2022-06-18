FROM python:3.9-slim

RUN pip install -U pip
RUN pip install pipenv
WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "predict.py", "model.py", "./"]
RUN pipenv install --deploy --system
COPY ["models/","models/linreg.bin", "./"]
EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
