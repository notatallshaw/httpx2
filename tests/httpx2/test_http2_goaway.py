from collections.abc import Iterable

import hpack
import hyperframe.frame
import pytest

import httpcore2
import httpx2


@pytest.mark.anyio
async def test_get_retries_http2_connection_that_processed_no_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    buffers = [
        [
            hyperframe.frame.SettingsFrame().serialize(),
            hyperframe.frame.GoAwayFrame(last_stream_id=0, error_code=0).serialize(),
        ],
        [
            hyperframe.frame.SettingsFrame().serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode([(b":status", b"200")]),
                flags=["END_HEADERS", "END_STREAM"],
            ).serialize(),
        ],
    ]
    backends = iter(httpcore2.AsyncMockBackend(buffer, http2=True) for buffer in buffers)
    connections = 0

    async def connect_tcp(
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore2.SOCKET_OPTION] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        nonlocal connections
        connections += 1
        return await next(backends).connect_tcp(host, port, timeout, local_address, socket_options)

    # Keep the real default transport and pool, replacing only network connection.
    transport = httpx2.AsyncHTTPTransport(http2=True)
    monkeypatch.setattr(transport._pool._network_backend, "connect_tcp", connect_tcp)

    async with httpx2.AsyncClient(transport=transport) as client:
        response = await client.get("https://example.com/")

    assert response.status_code == 200
    assert response.content == b""
    assert connections == 2
