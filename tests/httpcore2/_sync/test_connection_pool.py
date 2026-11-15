import logging
from collections.abc import Iterator, Iterable
from typing import Any

import h2.settings
import hpack
import hyperframe.frame
import pytest
from tests.httpcore2 import concurrency
from typing_extensions import override

import httpcore2
from httpcore2._models import ByteStream


class RecordingHTTP2Stream(httpcore2.MockStream):
    def __init__(self, buffer: list[bytes]) -> None:
        super().__init__(buffer, http2=True)
        self.writes: list[bytes] = []

    @override
    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.writes.append(buffer)


def http2_requests(stream: RecordingHTTP2Stream) -> list[tuple[int, bytes]]:
    """Decode client writes to expose stream attempts and their actual bodies."""
    data = b"".join(stream.writes)
    preface = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
    assert data.startswith(preface)
    view = memoryview(data[len(preface) :])
    bodies: dict[int, bytearray] = {}
    while view:
        frame, length = hyperframe.frame.Frame.parse_frame_header(view[:9])
        frame.parse_body(view[9 : 9 + length])
        if isinstance(frame, hyperframe.frame.HeadersFrame):
            bodies[frame.stream_id] = bytearray()
        elif isinstance(frame, hyperframe.frame.DataFrame):
            bodies[frame.stream_id].extend(frame.data)
        view = view[9 + length :]
    return [(stream_id, bytes(body)) for stream_id, body in bodies.items()]


class HTTP2SequenceBackend(httpcore2.MockBackend):
    """Supply a finite script of real HTTP/2 streams to the actual pool."""

    def __init__(self, buffers: list[list[bytes]]) -> None:
        super().__init__([], http2=True)
        self.buffers = iter(buffers)
        self.streams: list[RecordingHTTP2Stream] = []

    @override
    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore2.SOCKET_OPTION] | None = None,
    ) -> httpcore2.NetworkStream:
        buffer = next(self.buffers)
        stream = RecordingHTTP2Stream(buffer)
        self.streams.append(stream)
        return stream


def zero_goaway_response() -> list[bytes]:
    return [
        hyperframe.frame.SettingsFrame().serialize(),
        hyperframe.frame.GoAwayFrame(last_stream_id=0, error_code=0).serialize(),
    ]


def successful_http2_response(stream_id: int = 1) -> list[bytes]:
    return [
        hyperframe.frame.SettingsFrame().serialize(),
        hyperframe.frame.HeadersFrame(
            stream_id=stream_id,
            data=hpack.Encoder().encode([(b":status", b"200")]),
            flags=["END_HEADERS", "END_STREAM"],
        ).serialize(),
    ]



def test_connection_pool_with_keepalive() -> None:
    """
    By default HTTP/1.1 requests should be returned to the connection pool.
    """
    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
    )

    with httpcore2.ConnectionPool(network_backend=network_backend, max_keepalive_connections=1) as pool:
        # Sending an initial request, which once complete will return to the pool, IDLE.
        with pool.stream("GET", "https://example.com/") as response:
            info = [repr(c) for c in pool.connections]
            assert info == ["<HTTPConnection ['https://example.com:443', HTTP/1.1, ACTIVE, Request Count: 1]>"]
            assert repr(pool) == "<ConnectionPool [Requests: 1 active, 0 queued | Connections: 1 active, 0 idle]>"
            response.read()

        assert response.status == 200
        assert response.content == b"Hello, world!"
        info = [repr(c) for c in pool.connections]
        assert info == ["<HTTPConnection ['https://example.com:443', HTTP/1.1, IDLE, Request Count: 1]>"]
        assert repr(pool) == "<ConnectionPool [Requests: 0 active, 0 queued | Connections: 0 active, 1 idle]>"

        # Sending a second request to the same origin will reuse the existing IDLE connection.
        with pool.stream("GET", "https://example.com/") as response:
            info = [repr(c) for c in pool.connections]
            assert info == ["<HTTPConnection ['https://example.com:443', HTTP/1.1, ACTIVE, Request Count: 2]>"]
            assert repr(pool) == "<ConnectionPool [Requests: 1 active, 0 queued | Connections: 1 active, 0 idle]>"
            response.read()

        assert response.status == 200
        assert response.content == b"Hello, world!"
        info = [repr(c) for c in pool.connections]
        assert info == ["<HTTPConnection ['https://example.com:443', HTTP/1.1, IDLE, Request Count: 2]>"]
        assert repr(pool) == "<ConnectionPool [Requests: 0 active, 0 queued | Connections: 0 active, 1 idle]>"

        # Sending a request to a different origin will not reuse the existing IDLE connection.
        with (
            pool.stream("GET", "http://example.com/") as response_1,
            pool.stream("GET", "http://example.com/") as response_2,
        ):
            info = [repr(c) for c in pool.connections]
            assert info == [
                "<HTTPConnection ['https://example.com:443', HTTP/1.1, IDLE, Request Count: 2]>",
                "<HTTPConnection ['http://example.com:80', HTTP/1.1, ACTIVE, Request Count: 1]>",
                "<HTTPConnection ['http://example.com:80', HTTP/1.1, ACTIVE, Request Count: 1]>",
            ]
            assert repr(pool) == "<ConnectionPool [Requests: 2 active, 0 queued | Connections: 2 active, 1 idle]>"
            response_1.read()
            response_2.read()

        assert response_1.status == 200
        assert response_1.content == b"Hello, world!"
        assert response_2.status == 200
        assert response_2.content == b"Hello, world!"
        info = [repr(c) for c in pool.connections]
        assert info == [
            "<HTTPConnection ['http://example.com:80', HTTP/1.1, IDLE, Request Count: 1]>",
        ]
        assert repr(pool) == "<ConnectionPool [Requests: 0 active, 0 queued | Connections: 0 active, 1 idle]>"



def test_connection_pool_with_close() -> None:
    """
    HTTP/1.1 requests that include a 'Connection: Close' header should
    not be returned to the connection pool.
    """
    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
    )

    with httpcore2.ConnectionPool(network_backend=network_backend) as pool:
        # Sending an initial request, which once complete will not return to the pool.
        with pool.stream("GET", "https://example.com/", headers={"Connection": "close"}) as response:
            info = [repr(c) for c in pool.connections]
            assert info == ["<HTTPConnection ['https://example.com:443', HTTP/1.1, ACTIVE, Request Count: 1]>"]
            response.read()

        assert response.status == 200
        assert response.content == b"Hello, world!"
        info = [repr(c) for c in pool.connections]
        assert info == []



def test_connection_pool_with_http2() -> None:
    """
    Test a connection pool with HTTP/2 requests.
    """
    network_backend = httpcore2.MockBackend(
        buffer=[
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
        ],
        http2=True,
    )

    with httpcore2.ConnectionPool(
        network_backend=network_backend,
    ) as pool:
        # Sending an initial request, which once complete will return to the pool, IDLE.
        response = pool.request("GET", "https://example.com/")
        assert response.status == 200
        assert response.content == b"Hello, world!"

        info = [repr(c) for c in pool.connections]
        assert info == ["<HTTPConnection ['https://example.com:443', HTTP/2, IDLE, Request Count: 1]>"]

        # Sending a second request to the same origin will reuse the existing IDLE connection.
        response = pool.request("GET", "https://example.com/")
        assert response.status == 200
        assert response.content == b"Hello, world!"

        info = [repr(c) for c in pool.connections]
        assert info == ["<HTTPConnection ['https://example.com:443', HTTP/2, IDLE, Request Count: 2]>"]



def test_connection_pool_with_http2_goaway() -> None:
    """
    Test a connection pool with HTTP/2 requests, that cleanly disconnects
    with a GoAway frame after the first request.
    """
    network_backend = httpcore2.MockBackend(
        buffer=[
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
            hyperframe.frame.GoAwayFrame(stream_id=0, error_code=0, last_stream_id=1).serialize(),
            b"",
        ],
        http2=True,
    )

    with httpcore2.ConnectionPool(
        network_backend=network_backend,
    ) as pool:
        # Sending an initial request, which once complete will return to the pool, IDLE.
        response = pool.request("GET", "https://example.com/")
        assert response.status == 200
        assert response.content == b"Hello, world!"

        info = [repr(c) for c in pool.connections]
        assert info == ["<HTTPConnection ['https://example.com:443', HTTP/2, IDLE, Request Count: 1]>"]

        # Sending a second request to the same origin will require a new connection.
        # The original connection has now been closed.
        response = pool.request("GET", "https://example.com/")
        assert response.status == 200
        assert response.content == b"Hello, world!"

        info = [repr(c) for c in pool.connections]
        assert info == [
            "<HTTPConnection ['https://example.com:443', HTTP/2, IDLE, Request Count: 1]>",
        ]



def test_trace_request() -> None:
    """
    The 'trace' request extension allows for a callback function to inspect the
    internal events that occur while sending a request.
    """
    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
    )

    called: list[str] = []

    def trace(name: str, kwargs: dict[str, Any]) -> None:
        called.append(name)

    with httpcore2.ConnectionPool(network_backend=network_backend) as pool:
        pool.request("GET", "https://example.com/", extensions={"trace": trace})

    assert called == [
        "connection.connect_tcp.started",
        "connection.connect_tcp.complete",
        "connection.start_tls.started",
        "connection.start_tls.complete",
        "http11.send_request_headers.started",
        "http11.send_request_headers.complete",
        "http11.send_request_body.started",
        "http11.send_request_body.complete",
        "http11.receive_response_headers.started",
        "http11.receive_response_headers.complete",
        "http11.receive_response_body.started",
        "http11.receive_response_body.complete",
        "http11.response_closed.started",
        "http11.response_closed.complete",
    ]



def test_debug_request(caplog: pytest.LogCaptureFixture) -> None:
    """
    The 'trace' request extension allows for a callback function to inspect the
    internal events that occur while sending a request.
    """
    caplog.set_level(logging.DEBUG)

    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
    )

    with httpcore2.ConnectionPool(network_backend=network_backend) as pool:
        pool.request("GET", "http://example.com/")

    assert caplog.record_tuples == [
        (
            "httpcore2.connection",
            logging.DEBUG,
            "connect_tcp.started host='example.com' port=80 local_address=None timeout=None socket_options=None",
        ),
        (
            "httpcore2.connection",
            logging.DEBUG,
            "connect_tcp.complete return_value=<httpcore2.MockStream>",
        ),
        (
            "httpcore2.http11",
            logging.DEBUG,
            "send_request_headers.started request=<Request [b'GET']>",
        ),
        ("httpcore2.http11", logging.DEBUG, "send_request_headers.complete"),
        (
            "httpcore2.http11",
            logging.DEBUG,
            "send_request_body.started request=<Request [b'GET']>",
        ),
        ("httpcore2.http11", logging.DEBUG, "send_request_body.complete"),
        (
            "httpcore2.http11",
            logging.DEBUG,
            "receive_response_headers.started request=<Request [b'GET']>",
        ),
        (
            "httpcore2.http11",
            logging.DEBUG,
            "receive_response_headers.complete return_value="
            "(b'HTTP/1.1', 200, b'OK', [(b'Content-Type', b'plain/text'), (b'Content-Length', b'13')])",
        ),
        (
            "httpcore2.http11",
            logging.DEBUG,
            "receive_response_body.started request=<Request [b'GET']>",
        ),
        ("httpcore2.http11", logging.DEBUG, "receive_response_body.complete"),
        ("httpcore2.http11", logging.DEBUG, "response_closed.started"),
        ("httpcore2.http11", logging.DEBUG, "response_closed.complete"),
        ("httpcore2.connection", logging.DEBUG, "close.started"),
        ("httpcore2.connection", logging.DEBUG, "close.complete"),
    ]



def test_connection_pool_with_http_exception() -> None:
    """
    HTTP/1.1 requests that result in an exception during the connection should
    not be returned to the connection pool.
    """
    network_backend = httpcore2.MockBackend([b"Wait, this isn't valid HTTP!"])

    called: list[str] = []

    def trace(name: str, kwargs: dict[str, Any]) -> None:
        called.append(name)

    with httpcore2.ConnectionPool(network_backend=network_backend) as pool:
        # Sending an initial request, which once complete will not return to the pool.
        with pytest.raises(httpcore2.RemoteProtocolError):
            pool.request("GET", "https://example.com/", extensions={"trace": trace})

        info = [repr(c) for c in pool.connections]
        assert info == []

    assert called == [
        "connection.connect_tcp.started",
        "connection.connect_tcp.complete",
        "connection.start_tls.started",
        "connection.start_tls.complete",
        "http11.send_request_headers.started",
        "http11.send_request_headers.complete",
        "http11.send_request_body.started",
        "http11.send_request_body.complete",
        "http11.receive_response_headers.started",
        "http11.receive_response_headers.failed",
        "http11.response_closed.started",
        "http11.response_closed.complete",
    ]



def test_connection_pool_with_connect_exception() -> None:
    """
    HTTP/1.1 requests that result in an exception during connection should not
    be returned to the connection pool.
    """

    class FailedConnectBackend(httpcore2.MockBackend):
        def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: float | None = None,
            local_address: str | None = None,
            socket_options: Iterable[httpcore2.SOCKET_OPTION] | None = None,
        ) -> httpcore2.NetworkStream:
            raise httpcore2.ConnectError("Could not connect")

    network_backend = FailedConnectBackend([])

    called: list[str] = []

    def trace(name: str, kwargs: dict[str, Any]) -> None:
        called.append(name)

    with httpcore2.ConnectionPool(network_backend=network_backend) as pool:
        # Sending an initial request, which once complete will not return to the pool.
        with pytest.raises(httpcore2.ConnectError):
            pool.request("GET", "https://example.com/", extensions={"trace": trace})

        info = [repr(c) for c in pool.connections]
        assert info == []

    assert called == [
        "connection.connect_tcp.started",
        "connection.connect_tcp.failed",
    ]



def test_connection_pool_with_immediate_expiry() -> None:
    """
    Connection pools with keepalive_expiry=0.0 should immediately expire
    keep alive connections.
    """
    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
    )

    with httpcore2.ConnectionPool(
        keepalive_expiry=0.0,
        network_backend=network_backend,
    ) as pool:
        # Sending an initial request, which once complete will not return to the pool.
        with pool.stream("GET", "https://example.com/") as response:
            info = [repr(c) for c in pool.connections]
            assert info == ["<HTTPConnection ['https://example.com:443', HTTP/1.1, ACTIVE, Request Count: 1]>"]
            response.read()

        assert response.status == 200
        assert response.content == b"Hello, world!"
        info = [repr(c) for c in pool.connections]
        assert info == []



def test_connection_pool_with_no_keepalive_connections_allowed() -> None:
    """
    When 'max_keepalive_connections=0' is used, IDLE connections should not
    be returned to the pool.
    """
    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
    )

    with httpcore2.ConnectionPool(max_keepalive_connections=0, network_backend=network_backend) as pool:
        # Sending an initial request, which once complete will not return to the pool.
        with pool.stream("GET", "https://example.com/") as response:
            info = [repr(c) for c in pool.connections]
            assert info == ["<HTTPConnection ['https://example.com:443', HTTP/1.1, ACTIVE, Request Count: 1]>"]
            response.read()

        assert response.status == 200
        assert response.content == b"Hello, world!"
        info = [repr(c) for c in pool.connections]
        assert info == []



def test_connection_pool_closes_idle_connection_for_different_origin() -> None:
    """
    When the pool is at 'max_connections' and an incoming request is for an
    origin with no reusable connection, an IDLE connection to a different
    origin is closed to make room for the new connection.
    """
    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
    )

    with httpcore2.ConnectionPool(network_backend=network_backend, max_connections=1) as pool:
        # An initial request to a.com leaves a single IDLE connection in the pool.
        response = pool.request("GET", "https://a.com/")
        assert response.status == 200
        info = [repr(c) for c in pool.connections]
        assert info == ["<HTTPConnection ['https://a.com:443', HTTP/1.1, IDLE, Request Count: 1]>"]

        # A request to b.com cannot reuse the a.com connection and the pool is full,
        # so the IDLE a.com connection is closed and replaced with a b.com connection.
        response = pool.request("GET", "https://b.com/")
        assert response.status == 200
        info = [repr(c) for c in pool.connections]
        assert info == ["<HTTPConnection ['https://b.com:443', HTTP/1.1, IDLE, Request Count: 1]>"]



def test_connection_pool_concurrency() -> None:
    """
    HTTP/1.1 requests made in concurrency must not ever exceed the maximum number
    of allowable connection in the pool.
    """
    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
    )

    def fetch(pool: httpcore2.ConnectionPool, domain: str, info_list: list[list[str]]) -> None:
        with pool.stream("GET", f"http://{domain}/") as response:
            info = [repr(c) for c in pool.connections]
            info_list.append(info)
            response.read()

    with httpcore2.ConnectionPool(max_connections=1, network_backend=network_backend) as pool:
        info_list: list[list[str]] = []
        with concurrency.open_nursery() as nursery:
            for domain in ["a.com", "b.com", "c.com", "d.com", "e.com"]:
                nursery.start_soon(fetch, pool, domain, info_list)

        for item in info_list:
            # Check that each time we inspected the connection pool, only a
            # single connection was established at any one time.
            assert len(item) == 1
            # Each connection was to a different host, and only sent a single
            # request on that connection.
            assert item[0] in [
                "<HTTPConnection ['http://a.com:80', HTTP/1.1, ACTIVE, Request Count: 1]>",
                "<HTTPConnection ['http://b.com:80', HTTP/1.1, ACTIVE, Request Count: 1]>",
                "<HTTPConnection ['http://c.com:80', HTTP/1.1, ACTIVE, Request Count: 1]>",
                "<HTTPConnection ['http://d.com:80', HTTP/1.1, ACTIVE, Request Count: 1]>",
                "<HTTPConnection ['http://e.com:80', HTTP/1.1, ACTIVE, Request Count: 1]>",
            ]



def test_connection_pool_concurrency_same_domain_closing() -> None:
    """
    HTTP/1.1 requests made in concurrency must not ever exceed the maximum number
    of allowable connection in the pool.
    """
    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"Connection: close\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
    )

    def fetch(pool: httpcore2.ConnectionPool, domain: str, info_list: list[list[str]]) -> None:
        with pool.stream("GET", f"https://{domain}/") as response:
            info = [repr(c) for c in pool.connections]
            info_list.append(info)
            response.read()

    with httpcore2.ConnectionPool(max_connections=1, network_backend=network_backend, http2=True) as pool:
        info_list: list[list[str]] = []
        with concurrency.open_nursery() as nursery:
            for domain in ["a.com", "a.com", "a.com", "a.com", "a.com"]:
                nursery.start_soon(fetch, pool, domain, info_list)

        for item in info_list:
            # Check that each time we inspected the connection pool, only a
            # single connection was established at any one time.
            assert len(item) == 1
            # Only a single request was sent on each connection.
            assert item[0] == "<HTTPConnection ['https://a.com:443', HTTP/1.1, ACTIVE, Request Count: 1]>"



def test_connection_pool_concurrency_same_domain_keepalive() -> None:
    """
    HTTP/1.1 requests made in concurrency must not ever exceed the maximum number
    of allowable connection in the pool.
    """
    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
        * 5
    )

    def fetch(pool: httpcore2.ConnectionPool, domain: str, info_list: list[list[str]]) -> None:
        with pool.stream("GET", f"https://{domain}/") as response:
            info = [repr(c) for c in pool.connections]
            info_list.append(info)
            response.read()

    with httpcore2.ConnectionPool(max_connections=1, network_backend=network_backend, http2=True) as pool:
        info_list: list[list[str]] = []
        with concurrency.open_nursery() as nursery:
            for domain in ["a.com", "a.com", "a.com", "a.com", "a.com"]:
                nursery.start_soon(fetch, pool, domain, info_list)

        for item in info_list:
            # Check that each time we inspected the connection pool, only a
            # single connection was established at any one time.
            assert len(item) == 1
            # The connection sent multiple requests.
            assert item[0] in [
                "<HTTPConnection ['https://a.com:443', HTTP/1.1, ACTIVE, Request Count: 1]>",
                "<HTTPConnection ['https://a.com:443', HTTP/1.1, ACTIVE, Request Count: 2]>",
                "<HTTPConnection ['https://a.com:443', HTTP/1.1, ACTIVE, Request Count: 3]>",
                "<HTTPConnection ['https://a.com:443', HTTP/1.1, ACTIVE, Request Count: 4]>",
                "<HTTPConnection ['https://a.com:443', HTTP/1.1, ACTIVE, Request Count: 5]>",
            ]

    assert repr(pool) == "<ConnectionPool [Requests: 0 active, 0 queued | Connections: 0 active, 0 idle]>"



def test_unsupported_protocol() -> None:
    with httpcore2.ConnectionPool() as pool:
        with pytest.raises(httpcore2.UnsupportedProtocol):
            pool.request("GET", "ftp://www.example.com/")

        with pytest.raises(httpcore2.UnsupportedProtocol):
            pool.request("GET", "://www.example.com/")



def test_connection_pool_closed_while_request_in_flight() -> None:
    """
    Closing a connection pool while a request/response is still in-flight
    should raise an error.
    """
    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
    )

    with httpcore2.ConnectionPool(
        network_backend=network_backend,
    ) as pool:
        # Send a request, and then close the connection pool while the
        # response has not yet been streamed.
        with pool.stream("GET", "https://example.com/") as response:
            pool.close()
            with pytest.raises(httpcore2.ReadError):
                response.read()



def test_connection_pool_timeout() -> None:
    """
    Ensure that exceeding max_connections can cause a request to timeout.
    """
    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
    )

    with httpcore2.ConnectionPool(network_backend=network_backend, max_connections=1) as pool:
        # Send a request to a pool that is configured to only support a single
        # connection, and then ensure that a second concurrent request
        # fails with a timeout.
        with pool.stream("GET", "https://example.com/"):
            with pytest.raises(httpcore2.PoolTimeout):
                extensions = {"timeout": {"pool": 0.0001}}
                pool.request("GET", "https://example.com/", extensions=extensions)



def test_connection_pool_timeout_zero() -> None:
    """
    A pool timeout of 0 shouldn't raise a PoolTimeout if there's
    no need to wait on a new connection.
    """
    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
    )

    # Use a pool timeout of zero.
    extensions = {"timeout": {"pool": 0}}

    # A connection pool configured to allow only one connection at a time.
    with httpcore2.ConnectionPool(network_backend=network_backend, max_connections=1) as pool:
        # Two consecutive requests with a pool timeout of zero.
        # Both succeed without raising a timeout.
        response = pool.request("GET", "https://example.com/", extensions=extensions)
        assert response.status == 200
        assert response.content == b"Hello, world!"

        response = pool.request("GET", "https://example.com/", extensions=extensions)
        assert response.status == 200
        assert response.content == b"Hello, world!"

    # A connection pool configured to allow only one connection at a time.
    with httpcore2.ConnectionPool(network_backend=network_backend, max_connections=1) as pool:
        # Two concurrent requests with a pool timeout of zero.
        # Only the first will succeed without raising a timeout.
        with pool.stream("GET", "https://example.com/", extensions=extensions) as response:
            # The first response hasn't yet completed.
            with pytest.raises(httpcore2.PoolTimeout):
                # So a pool timeout occurs.
                pool.request("GET", "https://example.com/", extensions=extensions)
            # The first response now completes.
            response.read()

        assert response.status == 200
        assert response.content == b"Hello, world!"



def test_http11_upgrade_connection() -> None:
    """
    HTTP "101 Switching Protocols" indicates an upgraded connection.

    We should return the response, so that the network stream
    may be used for the upgraded connection.

    https://httpwg.org/specs/rfc9110.html#status.101
    https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/101
    """
    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 101 Switching Protocols\r\n",
            b"Connection: upgrade\r\n",
            b"Upgrade: custom\r\n",
            b"\r\n",
            b"...",
        ]
    )

    called: list[str] = []

    def trace(name: str, kwargs: dict[str, Any]) -> None:
        called.append(name)

    with httpcore2.ConnectionPool(network_backend=network_backend, max_connections=1) as pool:
        with pool.stream(
            "GET",
            "wss://example.com/",
            headers={"Connection": "upgrade", "Upgrade": "custom"},
            extensions={"trace": trace},
        ) as response:
            assert response.status == 101
            network_stream = response.extensions["network_stream"]
            content = network_stream.read(max_bytes=1024)
            assert content == b"..."

    assert called == [
        "connection.connect_tcp.started",
        "connection.connect_tcp.complete",
        "connection.start_tls.started",
        "connection.start_tls.complete",
        "http11.send_request_headers.started",
        "http11.send_request_headers.complete",
        "http11.send_request_body.started",
        "http11.send_request_body.complete",
        "http11.receive_response_headers.started",
        "http11.receive_response_headers.complete",
        "http11.response_closed.started",
        "http11.response_closed.complete",
    ]



def test_connection_pool_assigns_released_connection_to_one_queued_request() -> None:
    """
    A released connection must be handed to exactly one queued request.

    Assigning it to every queued request wakes them all, only for all but one
    to fail with `ConnectionNotAvailable` and re-enter the queue, degrading
    quadratically with queue depth.
    """

    class CountingPool(httpcore2.ConnectionPool):
        assign_passes = 0

        def _assign_requests_to_connections(self) -> list[httpcore2.ConnectionInterface]:
            CountingPool.assign_passes += 1
            return super()._assign_requests_to_connections()

    network_backend = httpcore2.MockBackend(
        [
            b"HTTP/1.1 200 OK\r\n",
            b"Content-Type: plain/text\r\n",
            b"Content-Length: 13\r\n",
            b"\r\n",
            b"Hello, world!",
        ]
        * 10
    )

    def fetch(pool: httpcore2.ConnectionPool) -> None:
        with pool.stream("GET", "https://example.com/") as response:
            response.read()
        assert response.status == 200

    with CountingPool(max_connections=1, network_backend=network_backend) as pool:
        with concurrency.open_nursery() as nursery:
            for _ in range(10):
                nursery.start_soon(fetch, pool)

    # Exactly two passes per request: one when it is queued, one when it
    # releases its connection.
    assert CountingPool.assign_passes == 2 * 10



def test_connection_pool_multiplexes_idle_http2_connection_within_a_pass() -> None:
    """
    A burst of requests arriving while a warmed HTTP/2 connection is idle
    must be assigned to it immediately, not serialized behind the first
    request's reservation.
    """

    class QueueObservingPool(httpcore2.ConnectionPool):
        max_queued_after_pass = 0

        def _assign_requests_to_connections(self) -> list[httpcore2.ConnectionInterface]:
            closing = super()._assign_requests_to_connections()
            queued = sum(request.is_queued() for request in self._requests)
            QueueObservingPool.max_queued_after_pass = max(QueueObservingPool.max_queued_after_pass, queued)
            return closing

    def response_frames(stream_id: int) -> list[bytes]:
        return [
            hyperframe.frame.HeadersFrame(
                stream_id=stream_id,
                data=hpack.Encoder().encode([(b":status", b"200")]),
                flags=["END_HEADERS"],
            ).serialize(),
            hyperframe.frame.DataFrame(stream_id=stream_id, data=b"Hello, world!", flags=["END_STREAM"]).serialize(),
        ]

    network_backend = httpcore2.MockBackend(
        buffer=[
            hyperframe.frame.SettingsFrame().serialize(),
            *response_frames(1),
            *response_frames(3),
            *response_frames(5),
            *response_frames(7),
        ],
        http2=True,
    )

    def fetch(pool: httpcore2.ConnectionPool) -> None:
        response = pool.request("GET", "https://example.com/")
        assert response.status == 200

    with QueueObservingPool(network_backend=network_backend, max_connections=1, http2=True) as pool:
        # Warm the connection; it returns to the pool IDLE.
        fetch(pool)
        with concurrency.open_nursery() as nursery:
            for _ in range(3):
                nursery.start_soon(fetch, pool)

    assert QueueObservingPool.max_queued_after_pass == 0



@pytest.mark.parametrize(("method", "content"), [("GET", None), ("POST", b"payload")])
def test_http2_zero_goaway_retries_unprocessed_request(method: str, content: bytes | None) -> None:
    backend = HTTP2SequenceBackend([zero_goaway_response(), successful_http2_response()])

    with httpcore2.ConnectionPool(http2=True, network_backend=backend) as pool:
        response = pool.request(method, "https://example.com/", content=content)

    assert response.status == 200
    assert response.content == b""
    assert len(backend.streams) == 2
    assert all(stream._closed for stream in backend.streams)
    assert [http2_requests(stream) for stream in backend.streams] == [[(1, content or b"")], [(1, content or b"")]]



def test_http2_zero_goaway_stops_after_failed_automatic_retry() -> None:
    # A third connect would exceed the one automatic replay, not exhaust a timer.
    backend = HTTP2SequenceBackend([zero_goaway_response(), zero_goaway_response()])

    with httpcore2.ConnectionPool(http2=True, network_backend=backend) as pool:
        with pytest.raises(httpcore2.RemoteProtocolError):
            pool.request("GET", "https://example.com/")

    assert len(backend.streams) == 2
    assert all(stream._closed for stream in backend.streams)



@pytest.mark.parametrize("status", [b"100", b"103", b"200"])
@pytest.mark.parametrize("last_stream_id", [0, 1])
def test_http2_goaway_does_not_replay_progressed_response(status: bytes, last_stream_id: int) -> None:
    progressed = [
        hyperframe.frame.SettingsFrame().serialize(),
        hyperframe.frame.HeadersFrame(
            stream_id=1,
            data=hpack.Encoder().encode([(b":status", status)]),
            flags=["END_HEADERS"],
        ).serialize(),
        hyperframe.frame.GoAwayFrame(last_stream_id=last_stream_id, error_code=0).serialize(),
    ]
    backend = HTTP2SequenceBackend([progressed, successful_http2_response()])

    with httpcore2.ConnectionPool(http2=True, network_backend=backend) as pool:
        with pytest.raises(httpcore2.RemoteProtocolError):
            pool.request("POST", "https://example.com/", content=b"payload")

    assert len(backend.streams) == 1



@pytest.mark.parametrize("subclass", [False, True])
def test_http2_zero_goaway_does_not_replay_unknown_body(subclass: bool) -> None:
    class CustomBytes(ByteStream):
        __slots__ = ()

    def content() -> Iterator[bytes]:
        yield b"payload"

    backend = HTTP2SequenceBackend([zero_goaway_response(), successful_http2_response()])

    with httpcore2.ConnectionPool(http2=True, network_backend=backend) as pool:
        with pytest.raises(httpcore2.RemoteProtocolError):
            if subclass:
                request = httpcore2.Request(
                    "POST",
                    "https://example.com/",
                    headers=[(b"Host", b"example.com"), (b"Content-Length", b"7")],
                    content=CustomBytes(b"payload"),
                )
                pool.handle_request(request)
            else:
                pool.request("POST", "https://example.com/", content=content())

    assert len(backend.streams) == 1



def test_http2_zero_goaway_keeps_upload_flow_refusal() -> None:
    backend = HTTP2SequenceBackend([zero_goaway_response(), successful_http2_response()])

    with httpcore2.ConnectionPool(http2=True, network_backend=backend) as pool:
        with pytest.raises(httpcore2.RemoteProtocolError):
            # One byte beyond the initial HTTP/2 stream window enters flow waiting.
            pool.request("POST", "https://example.com/", content=b"x" * 65536)

    assert len(backend.streams) == 1



def test_http2_protocol_retry_budget_does_not_bound_assignment_races() -> None:
    assignment_races = 0

    def trace(name: str, info: dict[str, Any]) -> None:
        nonlocal assignment_races
        if name == "connection.connect_tcp.started" and assignment_races < 2:
            assignment_races += 1
            raise httpcore2.ConnectionNotAvailable()

    backend = HTTP2SequenceBackend([zero_goaway_response(), successful_http2_response()])
    with httpcore2.ConnectionPool(http2=True, network_backend=backend) as pool:
        response = pool.request("GET", "https://example.com/", extensions={"trace": trace})

    assert response.status == 200
    assert assignment_races == 2
    assert len(backend.streams) == 2



@pytest.mark.parametrize("streaming", [False, True])
def test_http2_positive_goaway_replays_only_repeatable_bodies(streaming: bool) -> None:
    warmed = [*successful_http2_response(), hyperframe.frame.GoAwayFrame(last_stream_id=1, error_code=0).serialize()]
    backend = HTTP2SequenceBackend([warmed, successful_http2_response()])

    def content() -> Iterator[bytes]:
        yield b"payload"

    with httpcore2.ConnectionPool(http2=True, network_backend=backend) as pool:
        first = pool.request("GET", "https://example.com/")
        assert first.status == 200

        if streaming:
            with pytest.raises(httpcore2.RemoteProtocolError):
                pool.request("POST", "https://example.com/next", content=content())
        else:
            response = pool.request("POST", "https://example.com/next", content=b"payload")
            assert response.status == 200

    assert len(backend.streams) == (1 if streaming else 2)



@pytest.mark.parametrize(("method", "content"), [("GET", None), ("POST", b"payload")])
def test_http2_refused_stream_retries_on_the_same_connection(method: str, content: bytes | None) -> None:
    frames = [
        hyperframe.frame.SettingsFrame().serialize(),
        hyperframe.frame.RstStreamFrame(stream_id=1, error_code=7).serialize(),
        *successful_http2_response(stream_id=3)[1:],
    ]
    backend = HTTP2SequenceBackend([frames])

    with httpcore2.ConnectionPool(http2=True, network_backend=backend) as pool:
        response = pool.request(method, "https://example.com/", content=content)

    assert response.status == 200
    assert len(backend.streams) == 1
    assert http2_requests(backend.streams[0]) == [(1, content or b""), (3, content or b"")]



def test_http2_refused_stream_stops_after_failed_automatic_retry() -> None:
    backend = HTTP2SequenceBackend(
        [
            [
                hyperframe.frame.SettingsFrame().serialize(),
                hyperframe.frame.RstStreamFrame(stream_id=1, error_code=7).serialize(),
                hyperframe.frame.RstStreamFrame(stream_id=3, error_code=7).serialize(),
            ]
        ]
    )

    with httpcore2.ConnectionPool(http2=True, network_backend=backend) as pool:
        with pytest.raises(httpcore2.RemoteProtocolError):
            pool.request("GET", "https://example.com/")

    assert http2_requests(backend.streams[0]) == [(1, b""), (3, b"")]



@pytest.mark.parametrize(("status", "error_code"), [(None, 8), (b"100", 7), (b"103", 7), (b"200", 7)])
def test_http2_reset_only_replays_unprocessed_requests(status: bytes | None, error_code: int) -> None:
    frames = [hyperframe.frame.SettingsFrame().serialize()]
    if status is not None:
        frames.append(
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode([(b":status", status)]),
                flags=["END_HEADERS"],
            ).serialize()
        )
    frames.extend(
        [
            hyperframe.frame.RstStreamFrame(stream_id=1, error_code=error_code).serialize(),
            *successful_http2_response(stream_id=3)[1:],
        ]
    )
    backend = HTTP2SequenceBackend([frames])

    with httpcore2.ConnectionPool(http2=True, network_backend=backend) as pool:
        with pytest.raises(httpcore2.RemoteProtocolError):
            pool.request("POST", "https://example.com/", content=b"payload")

    assert http2_requests(backend.streams[0]) == [(1, b"payload")]



def test_http2_refused_stream_does_not_replay_a_consumed_iterator() -> None:
    frames = [
        hyperframe.frame.SettingsFrame().serialize(),
        hyperframe.frame.RstStreamFrame(stream_id=1, error_code=7).serialize(),
        *successful_http2_response(stream_id=3)[1:],
    ]
    backend = HTTP2SequenceBackend([frames])

    def content() -> Iterator[bytes]:
        yield b"payload"

    with httpcore2.ConnectionPool(http2=True, network_backend=backend) as pool:
        with pytest.raises(httpcore2.RemoteProtocolError):
            pool.request("POST", "https://example.com/", content=content())

    assert http2_requests(backend.streams[0]) == [(1, b"payload")]



def test_http2_goaway_retries_unprocessed_stream_beside_progressed_response() -> None:
    frames = [
        hyperframe.frame.SettingsFrame(settings={h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: 2}).serialize(),
        hyperframe.frame.HeadersFrame(
            stream_id=1,
            data=hpack.Encoder().encode([(b":status", b"200")]),
            flags=["END_HEADERS"],
        ).serialize(),
        hyperframe.frame.GoAwayFrame(last_stream_id=1, error_code=0).serialize(),
    ]
    backend = HTTP2SequenceBackend([frames, successful_http2_response()])

    with httpcore2.ConnectionPool(http2=True, network_backend=backend) as pool:
        with pool.stream("POST", "https://example.com/", content=b"payload") as first:
            assert first.status == 200
            second = pool.request("GET", "https://example.com/next")
            assert second.status == 200

    assert [http2_requests(stream) for stream in backend.streams] == [[(1, b"payload"), (3, b"")], [(1, b"")]]
