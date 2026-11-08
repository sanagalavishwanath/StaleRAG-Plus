from __future__ import annotations

from typing import Any


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, params: Any = None, headers: Any = None, timeout: Any = None) -> Any:
        self.calls.append({"url": url, "params": params})
        return self.handler(url, params or {})
