# End-to-End Machine Learning Deployment Pipeline

A clean, production-oriented machine learning project that shows how to train, package, and deploy a model as a web service using a modular, end-to-end pipeline.

The goal of this repository is to demonstrate how a real ML system is structured and deployed. A simple, well-known dataset is used only to keep the focus on engineering and system design rather than data collection.

## What This Project Shows

This project is built to reflect how machine learning is used in real applications:

- Creating a reusable ML pipeline
- Applying and saving preprocessing and trained models
- Serving predictions through a web API
- Keeping training and inference code separate
- Organizing ML projects for deployment rather than just experimentation
- Running the system in a container-friendly way

Together, these pieces form a practical example of how ML models move from training to production.

## Technology Stack

| Layer             | Tools                                |
| ----------------- | ------------------------------------ |
| Language          | Python 3                             |
| Data & ML         | pandas, numpy, scikit-learn, XGBoost |
| Model persistence | dill                                 |
| API & Web         | Flask                                |
| Validation        | Marshmallow                          |
| Visualization     | matplotlib, seaborn                  |
| Deployment        | Docker (planned)                     |

## System Architecture

```
User / Client
      |
      v
Flask Web App (app/)
      |
      v
Prediction Service
      |
      v
ML Inference Pipeline
      |
      v
Preprocessor  →  Model  →  Prediction
      |
      v
artifacts/
(preprocessor.pkl, model.pkl)
```

The web application never runs training code.  
It simply loads the saved preprocessing and model artifacts and uses them to generate predictions — which is how deployed ML systems typically operate.

## Project Structure

```
.
├── app/                    # Web layer
│   ├── routes/             # HTTP endpoints
│   ├── schemas/            # Input validation (Marshmallow)
│   ├── services/           # Prediction adapters
│   └── templates/          # Web UI
│
├── src/                    # ML pipeline layer
│   ├── components/         # Ingestion, transformation, training
│   ├── pipeline/           # Predict & (future) train pipelines
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
│
├── artifacts/              # Saved ML assets
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebook/               # Exploration & experiments
├── logs/
├── app.py                  # Flask entry point
└── requirements.txt
```

This structure allows:

- Training to run offline
- Inference to run independently
- Models or datasets to be swapped with minimal changes

## How Inference Works

When a user submits input:

1. Flask validates the input using Marshmallow
2. The data is converted into a DataFrame (`CustomData`)
3. `PredictPipeline` loads:
   - `artifacts/preprocessor.pkl`
   - `artifacts/model.pkl`
4. Preprocessing is applied
5. The model generates a prediction
6. A human-readable result is returned

No training code is executed during inference — this keeps the web service lightweight and reliable.

## Running Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Make sure model artifacts exist

The web app requires:

- `artifacts/model.pkl`
- `artifacts/preprocessor.pkl`

To generate them:

Place `raw_data.csv` in `notebook/data`, then run:

```bash
python src/components/data_ingestion.py
```

This will train the model and save all required artifacts.

### 3. Start the web server

```bash
python app.py
```

Open your browser at:

```
http://127.0.0.1:5000/
```

Enter sample inputs to receive live predictions from the model.

## Why This Architecture Matters

Many machine-learning projects stop at training a model in a notebook.  
This project goes a step further by showing how to package, load, and serve that model in a real application.

It demonstrates:

- Artifact-based inference
- API-driven prediction
- Clean separation between ML and web layers
- A structure that can scale beyond a single experiment

## Extensibility

The system is designed so that:

- The dataset can be replaced
- The model can be retrained
- The web app does not need to change
- CI/CD and versioning can be added later

This makes it a good foundation for building and evolving production-ready ML systems.

## Planned Improvements

- Dockerized deployment
- CI pipeline for training and validation
- Logging and monitoring
- Model versioning
- API-only (headless) inference mode

## Summary

This repository is a practical example of how to go from data to a deployed machine-learning service.  
It focuses on building a clean, maintainable pipeline that can be trained, packaged, and served in a way that mirrors real-world ML applications.
