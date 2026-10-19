"""API contract tests: the service must behave predictably at its edges."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def test_root_reports_service_and_model_state(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "sentiment"
    assert body["model_loaded"] is True


@pytest.mark.parametrize(
    "text, expected_sentiment, expected_label",
    [
        ("This movie was great", "positive", 1),
        ("I love every minute of it", "positive", 1),
        ("What a terrible film", "negative", 0),
        ("Boring and pointless", "negative", 0),
    ],
)
def test_predict_classifies_sentiment(client, text, expected_sentiment, expected_label):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == expected_sentiment
    assert body["prediction"] == expected_label
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_strips_surrounding_whitespace(client):
    response = client.post("/predict", json={"text": "   great movie   "})
    assert response.status_code == 200
    assert response.json()["text"] == "great movie"


@pytest.mark.parametrize(
    "payload",
    [
        {},  # missing field
        {"text": ""},  # empty string
        {"text": "      "},  # whitespace only
        {"text": None},  # null
        {"text": 123},  # wrong type
        {"text": "a" * 5001},  # above the max length
    ],
)
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_returns_503_when_no_model_is_available():
    no_model_client = TestClient(create_app(model=None))
    response = no_model_client.post("/predict", json={"text": "hello"})
    assert response.status_code == 503


def test_ui_page_is_served(client):
    response = client.get("/ui")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<!doctype html>" in response.text.lower()
