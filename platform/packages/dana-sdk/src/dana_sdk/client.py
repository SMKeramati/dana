"""Dana Python SDK client - OpenAI-compatible."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import httpx


class Dana:
    """کلاینت پایتون دانا.

    Usage:
        client = Dana(api_key="dk-...")
        response = client.chat("یک تابع بنویس")
        print(response)

        # استریم
        for token in client.chat_stream("داستان بنویس"):
            print(token, end="")
    """

    DEFAULT_BASE_URL = "https://api.dana.ir/v1"

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def chat(
        self,
        message: str,
        model: str = "qwen3-235b-moe",
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """ارسال پیام و دریافت پاسخ."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        response = self._client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def chat_stream(
        self,
        message: str,
        model: str = "qwen3-235b-moe",
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """ارسال پیام و دریافت پاسخ به صورت استریم."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        with self._client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    content = chunk["choices"][0].get("delta", {}).get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    def models(self) -> list[dict[str, Any]]:
        """دریافت لیست مدل‌های موجود."""
        response = self._client.get("/models")
        response.raise_for_status()
        return response.json()["data"]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Dana:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
