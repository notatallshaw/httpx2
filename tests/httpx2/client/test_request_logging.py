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


@pytest.mark.anyio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "extensions, expected_message",
    [
        ({}, 'HTTP Request: GET https://example.com/ "HTTP/1.1 200 OK"'),
        (
            {"http_version": b"HTTP/2", "reason_phrase": b"Custom reason"},
            'HTTP Request: GET https://example.com/ "HTTP/2 200 Custom reason"',
        ),
    ],
)
async def test_request_log_only_reads_properties_when_enabled(
    asynchronous: bool,
    enabled: bool,
    extensions: dict[str, bytes],
    expected_message: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = ReadCountingResponse(extensions)

    def respond(_request: httpx2.Request) -> httpx2.Response:
        return response

    transport = httpx2.MockTransport(respond)
    with caplog.at_level(logging.INFO if enabled else logging.WARNING, logger="httpx2"):
        if asynchronous:
            async with httpx2.AsyncClient(transport=transport) as async_client:
                received = await async_client.get("https://example.com/")
        else:
            with httpx2.Client(transport=transport) as sync_client:
                received = sync_client.get("https://example.com/")

    assert received is response
    assert received.content == b"body"
    messages = [record.getMessage() for record in caplog.records if record.name == "httpx2"]
    if enabled:
        assert response.property_reads == ["http_version", "reason_phrase"]
        assert messages == [expected_message]
    else:
        assert not response.property_reads
        assert not messages
