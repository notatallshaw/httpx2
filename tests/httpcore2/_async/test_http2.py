import h2.settings
import hpack
import hyperframe.frame
import pytest
import trio as concurrency

import httpcore2


def _response_bytes(stream_id: int, body: bytes = b"hello") -> bytes:
    """Encode a five-byte-declared response at the scripted I/O boundary."""
    return hyperframe.frame.HeadersFrame(
        stream_id=stream_id,
        data=hpack.Encoder().encode([(b":status", b"200"), (b"content-length", b"5")]),
        flags=["END_HEADERS"],
    ).serialize() + hyperframe.frame.DataFrame(stream_id=stream_id, data=body, flags=["END_STREAM"]).serialize()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure, expected",
    [
        ("malformed", httpcore2.LocalProtocolError),
        ("eof", httpcore2.RemoteProtocolError),
        ("read", httpcore2.ReadTimeout),
        ("write", httpcore2.WriteError),
    ],
)
async def test_http2_failed_connection_releases_pool_slot(
    monkeypatch: pytest.MonkeyPatch, failure: str, expected: type[Exception]
) -> None:
    backend = httpcore2.AsyncMockBackend(
        [hyperframe.frame.SettingsFrame().serialize(), _response_bytes(1)], http2=True
    )
    async with httpcore2.AsyncConnectionPool(
        network_backend=backend, max_connections=1, http1=False, http2=True
    ) as pool:
        first = await pool.request("GET", "http://example.com/first")
        assert first.content == b"hello"
        stream = first.extensions["network_stream"]
        assert isinstance(stream, httpcore2.AsyncMockStream)

        async def fail_read(max_bytes: int, timeout: float | None = None) -> bytes:
            raise httpcore2.ReadTimeout()

        async def fail_write(buffer: bytes, timeout: float | None = None) -> None:
            raise httpcore2.WriteError()

        if failure == "malformed":
            stream._buffer.append(_response_bytes(3, b"ab"))
        elif failure == "read":
            monkeypatch.setattr(stream, "read", fail_read)
        elif failure == "write":
            monkeypatch.setattr(stream, "write", fail_write)

        with pytest.raises(expected):
            await pool.request("GET", "http://example.com/failed")
        assert pool.connections == []
        assert stream._closed

        recovered = await pool.request(
            "GET", "http://example.com/recovered", extensions={"timeout": {"pool": 0.1}}
        )
        assert recovered.status == 200
        assert recovered.content == b"hello"
        assert len(pool.connections) == 1


@pytest.mark.anyio
async def test_http2_healthy_connection_remains_reusable() -> None:
    backend = httpcore2.AsyncMockBackend(
        [hyperframe.frame.SettingsFrame().serialize(), _response_bytes(1), _response_bytes(3)], http2=True
    )
    async with httpcore2.AsyncConnectionPool(
        network_backend=backend, max_connections=1, http1=False, http2=True
    ) as pool:
        first = await pool.request("GET", "http://example.com/first")
        connection = pool.connections[0]
        second = await pool.request("GET", "http://example.com/second")
        assert first.content == second.content == b"hello"
        assert pool.connections == [connection]
        assert connection.is_available()


@pytest.mark.anyio
async def test_http2_local_request_error_keeps_healthy_connection_open() -> None:
    stream = httpcore2.AsyncMockStream([hyperframe.frame.SettingsFrame().serialize()])
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as connection:
        with pytest.raises(httpcore2.LocalProtocolError):
            await connection.request("GET", "https://example.com/", headers=[(b":method", b"GET")])
        assert connection.is_available()
        assert not stream._closed


@pytest.mark.anyio
async def test_http2_exhausted_ids_close_after_last_response() -> None:
    last_stream_id = 2**31 - 1
    stream = httpcore2.AsyncMockStream(
        [
            hyperframe.frame.SettingsFrame(
                settings={h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: 2}
            ).serialize(),
            _response_bytes(last_stream_id),
        ]
    )
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as connection:
        # Exercise the RFC 9113 stream-ID boundary without billions of requests.
        connection._h2_state.highest_outbound_stream_id = last_stream_id - 2
        async with connection.stream("GET", "https://example.com/last") as response:
            with pytest.raises(httpcore2.ConnectionNotAvailable):
                await connection.request("GET", "https://example.com/exhausted")
            assert not connection.is_closed()
            assert await response.aread() == b"hello"
        assert connection.is_closed()
        assert stream._closed


@pytest.mark.anyio
async def test_http2_connection() -> None:
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    stream = httpcore2.AsyncMockStream(
        [
            hyperframe.frame.SettingsFrame().serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode(
                    [
                        (b":status", b"200"),
                        (b"content-type", b"plain/text"),
                    ]
                ),
                flags=["END_HEADERS"],
            ).serialize(),
            hyperframe.frame.DataFrame(stream_id=1, data=b"Hello, world!", flags=["END_STREAM"]).serialize(),
        ]
    )
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream, keepalive_expiry=5.0) as conn:
        response = await conn.request("GET", "https://example.com/")
        assert response.status == 200
        assert response.content == b"Hello, world!"

        assert conn.is_idle()
        assert conn.is_available()
        assert not conn.is_closed()
        assert not conn.has_expired()
        assert conn.is_connected()
        assert conn.info() == "'https://example.com:443', HTTP/2, IDLE, Request Count: 1"
        assert repr(conn) == "<AsyncHTTP2Connection ['https://example.com:443', IDLE, Request Count: 1]>"


@pytest.mark.anyio
async def test_http2_connection_closed() -> None:
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    stream = httpcore2.AsyncMockStream(
        [
            hyperframe.frame.SettingsFrame().serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode(
                    [
                        (b":status", b"200"),
                        (b"content-type", b"plain/text"),
                    ]
                ),
                flags=["END_HEADERS"],
            ).serialize(),
            hyperframe.frame.DataFrame(stream_id=1, data=b"Hello, world!", flags=["END_STREAM"]).serialize(),
            # Connection is closed after the first response
            hyperframe.frame.GoAwayFrame(stream_id=0, error_code=0, last_stream_id=1).serialize(),
        ]
    )
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream, keepalive_expiry=5.0) as conn:
        await conn.request("GET", "https://example.com/")

        with pytest.raises(httpcore2.ConnectionNotAvailable):
            await conn.request("GET", "https://example.com/")

        assert not conn.is_available()


@pytest.mark.anyio
async def test_http2_response_closed_twice() -> None:
    """
    Closing a response for a stream that has already been removed should be
    a no-op, rather than raising a `KeyError` that masks the exception which
    triggered the cleanup. See https://github.com/encode/httpx/issues/3072
    """
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    stream = httpcore2.AsyncMockStream(
        [
            hyperframe.frame.SettingsFrame().serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode(
                    [
                        (b":status", b"200"),
                        (b"content-type", b"plain/text"),
                    ]
                ),
                flags=["END_HEADERS"],
            ).serialize(),
            hyperframe.frame.DataFrame(stream_id=1, data=b"Hello, world!", flags=["END_STREAM"]).serialize(),
        ]
    )
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream, keepalive_expiry=5.0) as conn:
        await conn.request("GET", "https://example.com/")

        # The stream was closed when the response completed.
        await conn._response_closed(stream_id=1)


@pytest.mark.anyio
async def test_http2_connection_post_request() -> None:
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    stream = httpcore2.AsyncMockStream(
        [
            hyperframe.frame.SettingsFrame().serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode(
                    [
                        (b":status", b"200"),
                        (b"content-type", b"plain/text"),
                    ]
                ),
                flags=["END_HEADERS"],
            ).serialize(),
            hyperframe.frame.DataFrame(stream_id=1, data=b"Hello, world!", flags=["END_STREAM"]).serialize(),
        ]
    )
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as conn:
        response = await conn.request(
            "POST",
            "https://example.com/",
            headers={b"content-length": b"17"},
            content=b'{"data": "upload"}',
        )
        assert response.status == 200
        assert response.content == b"Hello, world!"


@pytest.mark.anyio
async def test_http2_connection_with_remote_protocol_error() -> None:
    """
    If a remote protocol error occurs, then no response will be returned,
    and the connection will not be reusable.
    """
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    stream = httpcore2.AsyncMockStream([b"Wait, this isn't valid HTTP!", b""])
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as conn:
        with pytest.raises(httpcore2.RemoteProtocolError):
            await conn.request("GET", "https://example.com/")


@pytest.mark.anyio
async def test_http2_connection_with_rst_stream() -> None:
    """
    If a stream reset occurs, then no response will be returned,
    but the connection will remain reusable for other requests.
    """
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    stream = httpcore2.AsyncMockStream(
        [
            hyperframe.frame.SettingsFrame().serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode(
                    [
                        (b":status", b"200"),
                        (b"content-type", b"plain/text"),
                    ]
                ),
                flags=["END_HEADERS"],
            ).serialize(),
            # Stream is closed midway through the first response...
            hyperframe.frame.RstStreamFrame(stream_id=1, error_code=8).serialize(),
            # ...Which doesn't prevent the second response.
            hyperframe.frame.HeadersFrame(
                stream_id=3,
                data=hpack.Encoder().encode(
                    [
                        (b":status", b"200"),
                        (b"content-type", b"plain/text"),
                    ]
                ),
                flags=["END_HEADERS"],
            ).serialize(),
            hyperframe.frame.DataFrame(stream_id=3, data=b"Hello, world!", flags=["END_STREAM"]).serialize(),
            b"",
        ]
    )
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as conn:
        with pytest.raises(httpcore2.RemoteProtocolError):
            await conn.request("GET", "https://example.com/")
        response = await conn.request("GET", "https://example.com/")
        assert response.status == 200


@pytest.mark.anyio
async def test_http2_connection_with_goaway() -> None:
    """
    If a GoAway frame occurs, then no response will be returned,
    and the connection will not be reusable for other requests.
    """
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    stream = httpcore2.AsyncMockStream(
        [
            hyperframe.frame.SettingsFrame().serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode(
                    [
                        (b":status", b"200"),
                        (b"content-type", b"plain/text"),
                    ]
                ),
                flags=["END_HEADERS"],
            ).serialize(),
            # Connection is closed midway through the first response...
            hyperframe.frame.GoAwayFrame(stream_id=0, error_code=0).serialize(),
            # ...We'll never get to this second response.
            hyperframe.frame.HeadersFrame(
                stream_id=3,
                data=hpack.Encoder().encode(
                    [
                        (b":status", b"200"),
                        (b"content-type", b"plain/text"),
                    ]
                ),
                flags=["END_HEADERS"],
            ).serialize(),
            hyperframe.frame.DataFrame(stream_id=3, data=b"Hello, world!", flags=["END_STREAM"]).serialize(),
            b"",
        ]
    )
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as conn:
        # The initial request has been closed midway, with an unrecoverable error.
        with pytest.raises(httpcore2.RemoteProtocolError):
            await conn.request("GET", "https://example.com/")

        # The second request can receive a graceful `ConnectionNotAvailable`,
        # and may be retried on a new connection.
        with pytest.raises(httpcore2.ConnectionNotAvailable):
            await conn.request("GET", "https://example.com/")


@pytest.mark.anyio
async def test_http2_connection_with_negative_flow_control_window() -> None:
    """A negative stream flow-control window must be awaited, not sent into.

    After the 65535-byte window is exhausted, the server reduces INITIAL_WINDOW_SIZE
    by 32767, which adjusts the just-exhausted stream window from 0 to -32767.
    `_wait_for_outgoing_flow` must park the stream until WINDOW_UPDATE restores
    positive credit; otherwise `h2` raises `LocalProtocolError` on the next send_data.
    """
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    reduce_settings = hyperframe.frame.SettingsFrame(stream_id=0)
    reduce_settings.settings = {hyperframe.frame.SettingsFrame.INITIAL_WINDOW_SIZE: 32768}
    stream = httpcore2.AsyncMockStream(
        [
            hyperframe.frame.SettingsFrame(stream_id=0).serialize(),
            # This frame reduces INITIAL_WINDOW_SIZE to 32768, which adjusts the just-exhausted stream window to -32767.
            reduce_settings.serialize(),
            hyperframe.frame.WindowUpdateFrame(stream_id=0, window_increment=100_000).serialize(),
            hyperframe.frame.WindowUpdateFrame(stream_id=1, window_increment=100_000).serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode(
                    [
                        (b":status", b"200"),
                        (b"content-type", b"plain/text"),
                    ]
                ),
                flags=["END_HEADERS"],
            ).serialize(),
            hyperframe.frame.DataFrame(stream_id=1, data=b"response", flags=["END_STREAM"]).serialize(),
        ]
    )
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as conn:
        response = await conn.request("POST", "https://example.com/", content=b"x" * 100_000)
        assert response.status == 200
        assert response.content == b"response"


@pytest.mark.anyio
async def test_http2_connection_with_flow_control() -> None:
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    stream = httpcore2.AsyncMockStream(
        [
            hyperframe.frame.SettingsFrame().serialize(),
            # Available flow: 65,535
            hyperframe.frame.WindowUpdateFrame(stream_id=0, window_increment=10_000).serialize(),
            hyperframe.frame.WindowUpdateFrame(stream_id=1, window_increment=10_000).serialize(),
            # Available flow: 75,535
            hyperframe.frame.WindowUpdateFrame(stream_id=0, window_increment=10_000).serialize(),
            hyperframe.frame.WindowUpdateFrame(stream_id=1, window_increment=10_000).serialize(),
            # Available flow: 85,535
            hyperframe.frame.WindowUpdateFrame(stream_id=0, window_increment=10_000).serialize(),
            hyperframe.frame.WindowUpdateFrame(stream_id=1, window_increment=10_000).serialize(),
            # Available flow: 95,535
            hyperframe.frame.WindowUpdateFrame(stream_id=0, window_increment=10_000).serialize(),
            hyperframe.frame.WindowUpdateFrame(stream_id=1, window_increment=10_000).serialize(),
            # Available flow: 105,535
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode(
                    [
                        (b":status", b"200"),
                        (b"content-type", b"plain/text"),
                    ]
                ),
                flags=["END_HEADERS"],
            ).serialize(),
            hyperframe.frame.DataFrame(stream_id=1, data=b"100,000 bytes received", flags=["END_STREAM"]).serialize(),
        ]
    )
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as conn:
        response = await conn.request(
            "POST",
            "https://example.com/",
            content=b"x" * 100_000,
        )
        assert response.status == 200
        assert response.content == b"100,000 bytes received"


class AsyncSlowServerStream(httpcore2.AsyncMockStream):
    """
    A mock stream where reads yield to the scheduler, as a socket read would,
    and the final chunk is only readable once two request streams have been opened.
    """

    def __init__(self, buffer: list[bytes]) -> None:
        super().__init__(buffer, http2=True)
        self.headers_frames_received = 0

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        while len(self._buffer) == 1 and self.headers_frames_received < 2:
            await concurrency.sleep(0)
        await concurrency.sleep(0)
        return await super().read(max_bytes, timeout)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        data = memoryview(buffer.removeprefix(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"))
        while data:
            frame, length = hyperframe.frame.Frame.parse_frame_header(data[:9])
            if isinstance(frame, hyperframe.frame.HeadersFrame):
                self.headers_frames_received += 1
            data = data[9 + length :]


@pytest.mark.trio
async def test_http2_connection_with_concurrent_uploads() -> None:
    """
    Two uploads exhaust the connection flow control window, and both wait for the
    server to extend it. A single read returns the WINDOW_UPDATE for both streams,
    together with the responses. The upload that was queued behind that read
    should not then block on a further read, since the server has nothing more to send.

    See https://github.com/pydantic/httpx2/issues/1133
    """
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    stream = AsyncSlowServerStream(
        [
            hyperframe.frame.SettingsFrame(settings={h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: 100}).serialize(),
            b"".join(
                [
                    hyperframe.frame.WindowUpdateFrame(stream_id=0, window_increment=2**20).serialize(),
                    hyperframe.frame.WindowUpdateFrame(stream_id=1, window_increment=2**20).serialize(),
                    hyperframe.frame.WindowUpdateFrame(stream_id=3, window_increment=2**20).serialize(),
                    hyperframe.frame.HeadersFrame(
                        stream_id=1, data=hpack.Encoder().encode([(b":status", b"200")]), flags=["END_HEADERS"]
                    ).serialize(),
                    hyperframe.frame.DataFrame(stream_id=1, data=b"", flags=["END_STREAM"]).serialize(),
                    hyperframe.frame.HeadersFrame(
                        stream_id=3, data=hpack.Encoder().encode([(b":status", b"200")]), flags=["END_HEADERS"]
                    ).serialize(),
                    hyperframe.frame.DataFrame(stream_id=3, data=b"", flags=["END_STREAM"]).serialize(),
                ]
            ),
        ]
    )
    # Each upload is larger than the 65,535 byte initial connection window.
    content = b"x" * 100_000

    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as conn:

        async def upload() -> None:
            response = await conn.request("POST", "https://example.com/", content=content)
            assert response.status == 200

        async with concurrency.open_nursery() as nursery:
            nursery.start_soon(upload)
            nursery.start_soon(upload)


@pytest.mark.anyio
async def test_http2_connection_attempt_close() -> None:
    """
    A connection can only be closed when it is idle.
    """
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    stream = httpcore2.AsyncMockStream(
        [
            hyperframe.frame.SettingsFrame().serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode(
                    [
                        (b":status", b"200"),
                        (b"content-type", b"plain/text"),
                    ]
                ),
                flags=["END_HEADERS"],
            ).serialize(),
            hyperframe.frame.DataFrame(stream_id=1, data=b"Hello, world!", flags=["END_STREAM"]).serialize(),
        ]
    )
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as conn:
        async with conn.stream("GET", "https://example.com/") as response:
            await response.aread()
            assert response.status == 200
            assert response.content == b"Hello, world!"

        await conn.aclose()
        with pytest.raises(httpcore2.ConnectionNotAvailable):
            await conn.request("GET", "https://example.com/")


@pytest.mark.anyio
async def test_http2_request_to_incorrect_origin() -> None:
    """
    A connection can only send requests to whichever origin it is connected to.
    """
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    stream = httpcore2.AsyncMockStream([])
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as conn:
        with pytest.raises(RuntimeError):
            await conn.request("GET", "https://other.com/")


@pytest.mark.anyio
async def test_http2_remote_max_streams_update() -> None:
    """
    If the remote server updates the maximum concurrent streams value, we should
    be adjusting how many streams we will allow.
    """
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    stream = httpcore2.AsyncMockStream(
        [
            hyperframe.frame.SettingsFrame(
                settings={hyperframe.frame.SettingsFrame.MAX_CONCURRENT_STREAMS: 1000}
            ).serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode(
                    [
                        (b":status", b"200"),
                        (b"content-type", b"plain/text"),
                    ]
                ),
                flags=["END_HEADERS"],
            ).serialize(),
            hyperframe.frame.DataFrame(stream_id=1, data=b"Hello, world!").serialize(),
            hyperframe.frame.SettingsFrame(
                settings={hyperframe.frame.SettingsFrame.MAX_CONCURRENT_STREAMS: 50}
            ).serialize(),
            hyperframe.frame.DataFrame(stream_id=1, data=b"Hello, world...again!", flags=["END_STREAM"]).serialize(),
        ]
    )
    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as conn:
        async with conn.stream("GET", "https://example.com/") as response:
            i = 0
            async for chunk in response.aiter_stream():
                if i == 0:
                    assert chunk == b"Hello, world!"
                    assert conn._h2_state.remote_settings.max_concurrent_streams == 1000
                    assert conn._max_streams == min(
                        conn._h2_state.remote_settings.max_concurrent_streams,
                        conn._h2_state.local_settings.max_concurrent_streams,
                    )
                elif i == 1:
                    assert chunk == b"Hello, world...again!"
                    assert conn._h2_state.remote_settings.max_concurrent_streams == 50
                    assert conn._max_streams == min(
                        conn._h2_state.remote_settings.max_concurrent_streams,
                        conn._h2_state.local_settings.max_concurrent_streams,
                    )
                i += 1


@pytest.mark.anyio
@pytest.mark.parametrize("limits", [[1], [1, 2], [1, 100], [1, 2, 1]])
async def test_http2_remote_max_streams_decrease_below_in_flight(limits: list[int]) -> None:
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    stream = httpcore2.AsyncMockStream(
        [
            hyperframe.frame.SettingsFrame(
                settings={hyperframe.frame.SettingsFrame.MAX_CONCURRENT_STREAMS: 100}
            ).serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode([(b":status", b"200")]),
                flags=["END_HEADERS"],
            ).serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=3,
                data=hpack.Encoder().encode([(b":status", b"200")]),
                flags=["END_HEADERS"],
            ).serialize(),
            *[
                hyperframe.frame.SettingsFrame(
                    settings={hyperframe.frame.SettingsFrame.MAX_CONCURRENT_STREAMS: limit}
                ).serialize()
                for limit in limits
            ],
            hyperframe.frame.DataFrame(stream_id=1, data=b"response a", flags=["END_STREAM"]).serialize(),
            hyperframe.frame.DataFrame(stream_id=3, data=b"response b", flags=["END_STREAM"]).serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=5,
                data=hpack.Encoder().encode([(b":status", b"200")]),
                flags=["END_HEADERS"],
            ).serialize(),
            hyperframe.frame.DataFrame(stream_id=5, data=b"response c", flags=["END_STREAM"]).serialize(),
        ]
    )

    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as conn:
        async with conn.stream("GET", "https://example.com/a") as response_a:
            async with conn.stream("GET", "https://example.com/b") as response_b:
                assert response_a.status == response_b.status == 200
                assert await response_a.aread() == b"response a"
                assert await response_b.aread() == b"response b"

        response_c = await conn.request("GET", "https://example.com/c")
        assert response_c.status == 200
        assert response_c.content == b"response c"
        assert conn.is_idle()
