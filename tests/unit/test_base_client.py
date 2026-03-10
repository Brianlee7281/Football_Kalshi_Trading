"""Tests for src/clients/base_client.py — retry logic, rate limiting, logging."""

from __future__ import annotations

import time
from unittest.mock import patch

import httpx
import pytest

from src.clients.base_client import BaseClient, RateLimitError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_transport() -> httpx.MockTransport:
    """Default transport that returns 200 OK."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handler)


@pytest.fixture
def client(mock_transport: httpx.MockTransport) -> BaseClient:
    c = BaseClient("https://api.example.com", max_retries=3, backoff_base=0.01)
    # Inject mock transport directly
    c._client = httpx.AsyncClient(
        base_url="https://api.example.com",
        transport=mock_transport,
    )
    return c


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_get_success(client: BaseClient) -> None:
    response = await client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    await client.close()


async def test_post_success(client: BaseClient) -> None:
    response = await client.post("/test", json={"key": "value"})
    assert response.status_code == 200
    await client.close()


# ---------------------------------------------------------------------------
# Retry on transient status codes
# ---------------------------------------------------------------------------


async def test_retry_on_500_then_success() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return httpx.Response(500, text="Server Error")
        return httpx.Response(200, json={"recovered": True})

    client = BaseClient("https://api.example.com", max_retries=3, backoff_base=0.001)
    client._client = httpx.AsyncClient(
        base_url="https://api.example.com",
        transport=httpx.MockTransport(handler),
    )

    response = await client.get("/retry-test")
    assert response.status_code == 200
    assert response.json() == {"recovered": True}
    assert call_count == 3  # 2 failures + 1 success
    await client.close()


async def test_retry_exhausted_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    client = BaseClient("https://api.example.com", max_retries=2, backoff_base=0.001)
    client._client = httpx.AsyncClient(
        base_url="https://api.example.com",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.get("/fail")
    assert exc_info.value.response.status_code == 503
    await client.close()


async def test_retry_on_429() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, text="Too Many Requests")
        return httpx.Response(200, json={"ok": True})

    client = BaseClient("https://api.example.com", max_retries=3, backoff_base=0.001)
    client._client = httpx.AsyncClient(
        base_url="https://api.example.com",
        transport=httpx.MockTransport(handler),
    )

    response = await client.get("/rate-limited")
    assert response.status_code == 200
    assert call_count == 2
    await client.close()


# ---------------------------------------------------------------------------
# Non-retryable errors (4xx except 429)
# ---------------------------------------------------------------------------


async def test_4xx_not_retried() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(404, text="Not Found")

    client = BaseClient("https://api.example.com", max_retries=3, backoff_base=0.001)
    client._client = httpx.AsyncClient(
        base_url="https://api.example.com",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await client.get("/missing")
    assert exc_info.value.response.status_code == 404
    assert call_count == 1  # No retries for 404
    await client.close()


# ---------------------------------------------------------------------------
# Retry on timeout
# ---------------------------------------------------------------------------


async def test_retry_on_timeout_then_success() -> None:
    call_count = 0

    async def mock_request(
        self: httpx.AsyncClient,
        method: str,
        url: str,
        **kwargs: object,
    ) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            raise httpx.ReadTimeout("Read timed out")
        return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, url))

    client = BaseClient("https://api.example.com", max_retries=3, backoff_base=0.001)
    client._client = httpx.AsyncClient(base_url="https://api.example.com")

    with patch.object(httpx.AsyncClient, "request", new=mock_request):
        response = await client.get("/timeout-test")
        assert response.status_code == 200
        assert call_count == 2

    await client.close()


async def test_timeout_exhausted_raises() -> None:
    async def mock_request(
        self: httpx.AsyncClient,
        method: str,
        url: str,
        **kwargs: object,
    ) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out")

    client = BaseClient("https://api.example.com", max_retries=2, backoff_base=0.001)
    client._client = httpx.AsyncClient(base_url="https://api.example.com")

    with (
        patch.object(httpx.AsyncClient, "request", new=mock_request),
        pytest.raises(httpx.ReadTimeout),
    ):
        await client.get("/timeout-fail")

    await client.close()


# ---------------------------------------------------------------------------
# Exponential backoff
# ---------------------------------------------------------------------------


def test_compute_backoff() -> None:
    client = BaseClient("https://example.com", backoff_base=1.0)
    assert client._compute_backoff(0) == 1.0  # 1 * 2^0
    assert client._compute_backoff(1) == 2.0  # 1 * 2^1
    assert client._compute_backoff(2) == 4.0  # 1 * 2^2
    assert client._compute_backoff(3) == 8.0  # 1 * 2^3


def test_compute_backoff_custom_base() -> None:
    client = BaseClient("https://example.com", backoff_base=0.5)
    assert client._compute_backoff(0) == 0.5
    assert client._compute_backoff(1) == 1.0
    assert client._compute_backoff(2) == 2.0


# ---------------------------------------------------------------------------
# Rate limit tracking
# ---------------------------------------------------------------------------


async def test_rate_limit_tracking(client: BaseClient) -> None:
    assert client.requests_in_last_minute == 0
    await client.get("/req1")
    assert client.requests_in_last_minute == 1
    await client.get("/req2")
    assert client.requests_in_last_minute == 2
    await client.close()


async def test_rate_limit_exceeded() -> None:
    client = BaseClient(
        "https://api.example.com",
        rate_limit_rpm=2,
        backoff_base=0.001,
    )
    client._client = httpx.AsyncClient(
        base_url="https://api.example.com",
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )

    await client.get("/r1")
    await client.get("/r2")

    with pytest.raises(RateLimitError):
        await client.get("/r3")

    await client.close()


async def test_rate_limit_zero_means_unlimited(client: BaseClient) -> None:
    assert client.rate_limit_rpm == 0
    for _ in range(10):
        await client.get("/unlimited")
    assert client.requests_in_last_minute == 10
    await client.close()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


async def test_context_manager() -> None:
    async with BaseClient("https://api.example.com") as client:
        assert client._client is not None
        assert not client._client.is_closed
    assert client._client is None


# ---------------------------------------------------------------------------
# Close idempotency
# ---------------------------------------------------------------------------


async def test_close_idempotent(client: BaseClient) -> None:
    await client.close()
    await client.close()  # Should not raise


# ---------------------------------------------------------------------------
# Request records timestamp correctly
# ---------------------------------------------------------------------------


def test_record_request_prunes_old() -> None:
    client = BaseClient("https://example.com")
    # Manually inject an old timestamp
    client._request_timestamps.append(time.monotonic() - 120)
    client._request_timestamps.append(time.monotonic())

    assert client.requests_in_last_minute == 1  # old one pruned
