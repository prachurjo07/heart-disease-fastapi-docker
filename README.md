# Heart Disease Prediction API

This project is a simple Machine Learning deployment assignment using FastAPI, Docker, and Render.

The API predicts whether a patient has heart disease based on clinical input features from the Heart Disease Dataset.

## Project Structure

```text
heart-disease-fastapi-docker/
│
├── app/
│   ├── main.py
│   └── schemas.py
│
├── model/
│   ├── train_model.py
│   └── heart_model.joblib
│
├── data/
│   └── heart.csv
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md