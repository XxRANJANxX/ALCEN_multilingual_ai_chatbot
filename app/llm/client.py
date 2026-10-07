
import json
import logging
from collections.abc import AsyncIterator

import httpx

from app.core.config import Settings

log = logging.getLogger(__name__)
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "host.docker.internal")


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, settings: Settings):
        self.base = settings.llm_base_url.rstrip("/")
        self.key = settings.llm_api_key
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout

    @property
    def available(self) -> bool:
        return bool(self.key) or any(h in self.base for h in _LOCAL_HOSTS)

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        return h


    def _payload(self, messages, temperature, max_tokens, stream=False) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    async def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024) -> str:
        if not self.available:
            raise LLMError("LLM not configured (set LLM_API_KEY or use a local endpoint)")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                r = await c.post(
                    f"{self.base}/chat/completions",
                    headers=self._headers(),
                    json=self._payload(messages, temperature, max_tokens),
                )
            if r.status_code >= 400:
                raise LLMError(f"{r.status_code}: {r.text[:300]}")
            return r.json()["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise LLMError(str(e)) from e

    async def stream(
        self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024
    ) -> AsyncIterator[str]:
        if not self.available:
            raise LLMError("LLM not configured (set LLM_API_KEY or use a local endpoint)")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                async with c.stream(
                    "POST",
                    f"{self.base}/chat/completions",
                    headers=self._headers(),
                    json=self._payload(messages, temperature, max_tokens, stream=True),
                ) as r:
                    if r.status_code >= 400:
                        body = (await r.aread()).decode("utf-8", "ignore")[:300]
                        raise LLMError(f"{r.status_code}: {body}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            delta = json.loads(payload)["choices"][0]["delta"].get("content")
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                        if delta:
                            yield delta
        except httpx.HTTPError as e:
            raise LLMError(str(e)) from e
