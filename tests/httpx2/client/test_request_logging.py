import logging

import pytest
from typing_extensions import override

import httpx2


class ReadCountingResponse(httpx2.Response):
    """Observe logging properties while retaining their actual decoding."""

    def __init__(self, extensions: dict[str, bytes]) -> None:
        self.property_reads: list[str] = []
        super().__init__(
            200,
            content=b"body",
            extensions=extensions,
        )

    @property
    @override
    def http_version(self) -> str:
        self.property_reads.append("http_version")
        return super().http_version

    @property
    @override
    def reason_phrase(self) -> str:
        self.property_reads.append("reason_phrase")
        return super().reason_phrase


@pytest.fixture(params=[{}, {"http_version": b"HTTP/2", "reason_phrase": b"Custom reason"}])
def response(request: pytest.FixtureRequest) -> ReadCountingResponse:
    return ReadCountingResponse(request.param)


def check_request_log(
    response: ReadCountingResponse,
    caplog: pytest.LogCaptureFixture,
    *,
    enabled: bool,
) -> None:
    """Check decoded fields and the resulting request message together."""
    messages = [record.getMessage() for record in caplog.records if record.name == "httpx2"]
    if enabled:
        assert response.property_reads == ["http_version", "reason_phrase"]
        expected = 'HTTP Request: GET https://example.com/ "HTTP/1.1 200 OK"'
        if response.extensions:
            expected = 'HTTP Request: GET https://example.com/ "HTTP/2 200 Custom reason"'
        assert messages == [expected]
    else:
        assert not response.property_reads
        assert not messages


@pytest.mark.parametrize("enabled", [False, True])
def test_request_log_only_reads_properties_when_enabled(
    response: ReadCountingResponse,
    enabled: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = httpx2.MockTransport(lambda request: response)
    with caplog.at_level(logging.INFO if enabled else logging.WARNING, logger="httpx2"):
        with httpx2.Client(transport=transport) as client:
            received = client.get("https://example.com/")

    assert received is response
    assert received.content == b"body"
    check_request_log(response, caplog, enabled=enabled)


@pytest.mark.anyio
@pytest.mark.parametrize("enabled", [False, True])
async def test_async_request_log_only_reads_properties_when_enabled(
    response: ReadCountingResponse,
    enabled: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    transport = httpx2.MockTransport(lambda request: response)
    with caplog.at_level(logging.INFO if enabled else logging.WARNING, logger="httpx2"):
        async with httpx2.AsyncClient(transport=transport) as client:
            received = await client.get("https://example.com/")

    assert received is response
    assert received.content == b"body"
    check_request_log(response, caplog, enabled=enabled)
