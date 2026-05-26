"""Tests for Dana SDK."""

from dana_sdk import Dana


class TestDanaClient:
    def test_init(self) -> None:
        client = Dana(api_key="dk-test123", base_url="http://localhost:8000/v1")
        assert client.api_key == "dk-test123"
        assert client.base_url == "http://localhost:8000/v1"
        client.close()

    def test_default_base_url(self) -> None:
        client = Dana(api_key="dk-test")
        assert "dana.ir" in client.base_url
        client.close()

    def test_context_manager(self) -> None:
        with Dana(api_key="dk-test", base_url="http://localhost:8000/v1") as client:
            assert client.api_key == "dk-test"
