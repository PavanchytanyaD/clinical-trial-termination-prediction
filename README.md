# Predicting Early Termination of Clinical Trials

MLOps course project (Northeastern University). The system predicts the probability that a registered clinical trial will be stopped early (terminated, withdrawn, or suspended) from its design, and an agentic AI system explains why a flagged trial is at risk by citing similar past trials that were stopped.

## Approach

- **Data:** ClinicalTrials.gov, accessed through the AACT database (monthly PostgreSQL snapshot). One row is one trial, identified by its NCT ID.
- **Baseline:** a classical ML model (gradient-boosted trees or logistic regression) on the structured design features.
- **Main model:** BioBERT text embeddings (not fine-tuned) combined with the structured features in a neural network with a classification head.
- **Agentic system:** looks up similar past stopped trials in the structured data and a text index, reads their recorded stop reasons, and writes a cited explanation.
- **Platform:** Google Cloud (Vertex AI Pipelines, BigQuery, Cloud Storage, Artifact Registry, GKE, Vertex AI Vector Search), with GitHub Actions, DVC, and MLflow.

## Repository structure

```
clinical-trial-termination-prediction/
|- data/        raw and processed data (DVC-tracked, not committed)
|- src/         ingestion, features, model, and agent code
|- pipelines/   pipeline definitions (ingest, train, evaluate, deploy)
|- tests/       unit and integration tests
|- docs/        project scoping document and diagrams
|- README.md
```

## Architecture

Training pipeline:

![Training pipeline](docs/diagrams/2_gcp_training.png)

Inference pipeline:

![Inference pipeline](docs/diagrams/3_gcp_inference.png)

## Installation (tentative)

These steps are a plan and will be finalized as the code is built.

**Prerequisites:** Python 3.10 or later, Git, Docker, DVC, and the Google Cloud CLI with access to the project's GCP account.

1. Clone the repository:
   ```bash
   git clone https://github.com/PavanchytanyaD/clinical-trial-termination-prediction.git
   cd clinical-trial-termination-prediction
   ```
2. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Sign in to Google Cloud:
   ```bash
   gcloud auth application-default login
   ```
4. Pull the versioned data:
   ```bash
   dvc pull
   ```

## Usage (tentative)

These guidelines are a plan and will be updated with the exact commands as each pipeline is built.

- **Data pipeline:** downloads the monthly AACT snapshot, cleans it, builds the structured features and BioBERT embeddings, and writes them to BigQuery.
- **Training pipeline:** runs on Vertex AI Pipelines on a set schedule or when a new data version arrives. It trains the baseline and the neural network, evaluates them, and approves a model only if it passes the evaluation gate.
- **Inference:** a FastAPI service on GKE takes a trial (NCT ID), returns the probability that it stops early, and gives a short explanation that cites similar past trials.
- **Tests:** run with `pytest` from the repository root.

## Team

- Atharva Nilesh Mahadik
- Danush Gopinath
- Nithish Bhat
- Pavanchytanya Dharmapuri
- Sathyasai Rajan
- Sivapriya Sugumaran

## Data rights

ClinicalTrials.gov and AACT (CTTI) data are public. The data is trial-level and contains no patient-level personal data.
