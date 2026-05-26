"""Tests for API gateway routes."""

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


class TestHealth:
    def test_health_endpoint(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["service"] == "api-gateway"


class TestModels:
    def test_list_models(self) -> None:
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == "qwen3-235b-moe"

    def test_get_model(self) -> None:
        response = client.get("/v1/models/qwen3-235b-moe")
        assert response.status_code == 200
        assert response.json()["id"] == "qwen3-235b-moe"


class TestChatCompletions:
    def test_chat_completions_no_auth(self) -> None:
        """Without auth, should return 401."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-235b-moe",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 401
