"""Shared async HTTP client with retry logic, rate tracking, and structured logging."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import httpx

from src.common.logging import get_logger


class BaseClient:
    """httpx.AsyncClient wrapper with retry, rate-limit tracking, and logging.

    Args:
        base_url: API base URL.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts on transient failures.
        backoff_base: Base delay (seconds) for exponential backoff.
        rate_limit_rpm: Requests-per-minute cap. 0 means unlimited.
    """

    RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        rate_limit_rpm: int = 0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.rate_limit_rpm = rate_limit_rpm

        self._client: httpx.AsyncClient | None = None
        self._headers = headers or {}
        self._request_timestamps: deque[float] = deque()
        self._logger = get_logger("base_client", base_url=base_url)

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self._headers,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Rate-limit tracking
    # ------------------------------------------------------------------

    def _record_request(self) -> None:
        """Record a request timestamp for RPM tracking."""
        now = time.monotonic()
        self._request_timestamps.append(now)
        # Prune timestamps older than 60s
        cutoff = now - 60.0
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()

    @property
    def requests_in_last_minute(self) -> int:
        """Return the number of requests made in the last 60 seconds."""
        now = time.monotonic()
        cutoff = now - 60.0
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()
        return len(self._request_timestamps)

    def _check_rate_limit(self) -> bool:
        """Return True if a request is allowed under the RPM cap."""
        if self.rate_limit_rpm <= 0:
            return True
        return self.requests_in_last_minute < self.rate_limit_rpm

    # ------------------------------------------------------------------
    # Core request with retry
    # ------------------------------------------------------------------

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Make an HTTP request with exponential-backoff retry on transient errors.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: URL path relative to base_url.
            params: Query parameters.
            json: JSON body payload.
            headers: Additional headers for this request.

        Returns:
            The httpx.Response on success.

        Raises:
            httpx.HTTPStatusError: On non-retryable 4xx/5xx after exhausting retries.
            httpx.TimeoutException: If all retries time out.
            RateLimitError: If the RPM cap is hit.
        """
        if not self._check_rate_limit():
            self._logger.warning(
                "rate_limit_exceeded",
                rpm=self.requests_in_last_minute,
                cap=self.rate_limit_rpm,
            )
            raise RateLimitError(
                f"Rate limit exceeded: {self.requests_in_last_minute}/{self.rate_limit_rpm} RPM"
            )

        client = await self._ensure_client()
        last_exception: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                self._record_request()
                self._logger.debug(
                    "request_start",
                    method=method,
                    path=path,
                    attempt=attempt + 1,
                )

                response = await client.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    headers=headers,
                )

                self._logger.debug(
                    "request_complete",
                    method=method,
                    path=path,
                    status=response.status_code,
                    attempt=attempt + 1,
                )

                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    last_exception = httpx.HTTPStatusError(
                        f"Server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                    if attempt < self.max_retries:
                        delay = self._compute_backoff(attempt)
                        self._logger.warning(
                            "retryable_status",
                            status=response.status_code,
                            attempt=attempt + 1,
                            delay=delay,
                        )
                        await _sleep(delay)
                        continue
                    # Final attempt — raise
                    response.raise_for_status()

                response.raise_for_status()
                return response

            except httpx.TimeoutException as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    delay = self._compute_backoff(attempt)
                    self._logger.warning(
                        "timeout_retry",
                        path=path,
                        attempt=attempt + 1,
                        delay=delay,
                    )
                    await _sleep(delay)
                    continue

            except httpx.HTTPStatusError:
                raise

            except httpx.HTTPError as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    delay = self._compute_backoff(attempt)
                    self._logger.warning(
                        "connection_error_retry",
                        path=path,
                        attempt=attempt + 1,
                        error=str(exc),
                        delay=delay,
                    )
                    await _sleep(delay)
                    continue

        # All retries exhausted
        self._logger.error(
            "request_failed",
            method=method,
            path=path,
            retries=self.max_retries,
        )
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Unexpected: no exception captured after retry exhaustion")

    def _compute_backoff(self, attempt: int) -> float:
        """Compute exponential backoff delay for a given attempt (0-indexed)."""
        return float(self.backoff_base * (2 ** attempt))

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    async def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Convenience GET request."""
        return await self.request("GET", path, params=params, headers=headers)

    async def post(
        self,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Convenience POST request."""
        return await self.request("POST", path, json=json, params=params, headers=headers)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> BaseClient:
        await self._ensure_client()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


class RateLimitError(Exception):
    """Raised when the client-side RPM rate limit is exceeded."""


async def _sleep(seconds: float) -> None:
    """Thin wrapper around asyncio.sleep for testability."""
    import asyncio

    await asyncio.sleep(seconds)
