# Deployment Guide

## Overview

The project uses GitHub Actions for CI/CD, Docker for packaging and DVC for
dataset versioning on Azure Blob Storage.

## Pipeline workflow

1. **Pull request** to `dev`, `staging` or `main`
   - checkout
   - install dependencies
   - `ruff check .`
   - `black --check .`
   - `pytest -q`
   - `docker build`
2. **Push to `dev`** — same checks; no deployment image.
3. **Push to `staging`**
   - run the same checks
   - create `.env` with `APP_ENV=staging`
   - build the `sentiment-app-staging` image
4. **Push to `main`**
   - run the same checks
   - create `.env` with `APP_ENV=production`
   - build the `sentiment-app-main` image

## Environment configuration

| Branch | APP_ENV | Secret |
|---|---|---|
| `staging` | `staging` | `STAGING_DVC_SAS_TOKEN` |
| `main` | `production` | `PROD_DVC_SAS_TOKEN` |

## Local deployment

1. Clone the repository.
2. `dvc pull`
3. Train the model: `python -m ml.train`
4. Build: `docker build -t sentiment-app .`
5. Run: `docker run -p 8000:8000 --env-file .env sentiment-app`

## Notes

- Raw data and model artifacts live in Azure Blob Storage through DVC.
- Never commit the SAS token; use GitHub Actions secrets.