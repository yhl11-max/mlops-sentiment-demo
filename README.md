# MLOps sentiment demo project

Coursework project for DevOps & MLOps: IMDB sentiment classification.

## Project structure

- `app/` — inference web application
- `ml/` — training, data pipeline and experiment tracking
- `tests/` — unit, integration and API tests
- `data/` — dataset tracked by DVC
- `.github/workflows/ci.yml` — CI/CD pipeline

## CI/CD pipeline

| Event | What runs |
|---|---|
| Pull request to `dev`, `staging`, `main` | ruff, black, pytest, Docker build |
| Push to `dev` | same checks; no deployment image |
| Push to `staging` | checks, then staging `.env` + staging image |
| Push to `main` | checks, then production `.env` + production image |

Required repository secrets:

- `STAGING_DVC_SAS_TOKEN`
- `PROD_DVC_SAS_TOKEN`

## How to run locally

1. Clone the repository.
2. Run `dvc pull` to fetch the dataset.
3. Train the model with `python -m ml.train`.
4. Build the image: `docker build -t sentiment-app .`
5. Run it: `docker run -p 8000:8000 --env-file .env sentiment-app`