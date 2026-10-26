"""Check admission while real HTTP/2 peers pause and resume new streams."""

from __future__ import annotations

from contextlib import AsyncExitStack

import anyio
import h2.config
import h2.connection
import h2.events
import h2.settings
import hpack
import hyperframe.frame
import pytest

import httpcore2


@pytest.mark.anyio
async def test_overlapping_stream_ids_keep_request_owned_capacity() -> None:
    """Delayed header traces must not merge separate requests' capacity or cleanup."""
    stream = httpcore2.AsyncMockStream(
        [
            hyperframe.frame.SettingsFrame(settings={h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: 2}).serialize(),
            hyperframe.frame.HeadersFrame(
                stream_id=1,
                data=hpack.Encoder().encode([(b":status", b"200")]),
                flags=["END_HEADERS", "END_STREAM"],
            ).serialize(),
        ],
        http2=True,
    )
    scopes = [anyio.CancelScope() for _ in range(4)]
    started = [anyio.Event() for _ in range(4)]
    done = [anyio.Event() for _ in range(4)]
    admitted: list[tuple[int, int]] = []
    origin = httpcore2.Origin(b"https", b"example.test", 443)

    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as connection:
        warm = await connection.request("GET", "https://example.test/warm")
        assert warm.status == 200
        assert warm.content == b""

        async def request(index: int) -> None:
            async def trace(name: str, info: dict[str, object]) -> None:
                if name == "http2.send_request_headers.started":
                    stream_id = info["stream_id"]
                    assert isinstance(stream_id, int)
                    admitted.append((index, stream_id))
                    started[index].set()
                    await anyio.sleep_forever()

            with scopes[index]:
                await connection.request("GET", "https://example.test/held", extensions={"trace": trace})
            done[index].set()

        with anyio.fail_after(2):
            async with anyio.create_task_group() as group:
                group.start_soon(request, 0)
                await started[0].wait()
                group.start_soon(request, 1)
                await started[1].wait()
                group.start_soon(request, 2)
                await anyio.wait_all_tasks_blocked()
                assert [index for index, _stream_id in admitted] == [0, 1]
                reservations_before = connection._stream_reservations.copy()

                scopes[0].cancel()
                await done[0].wait()
                await started[2].wait()
                first_reservation = (reservations_before - connection._stream_reservations).pop()
                assert [index for index, _stream_id in admitted] == [0, 1, 2]
                third_id = admitted[2][1]
                third_events = connection._events[third_id]

                group.start_soon(request, 3)
                await anyio.wait_all_tasks_blocked()
                assert [index for index, _stream_id in admitted] == [0, 1, 2]
                scopes[3].cancel()
                await done[3].wait()

                scopes[1].cancel()
                await done[1].wait()
                assert connection._events[third_id] is third_events
                assert not connection.is_idle()

                await connection._response_closed(stream_id=admitted[0][1], reservation=first_reservation)
                assert connection._events[third_id] is third_events
                assert not connection.is_idle()

                scopes[2].cancel()
                await done[2].wait()

        assert connection.is_idle()


class ControlledPeer(httpcore2.AsyncMockStream):
    """Keep response bodies under test control while h2 checks the wire state."""

    def __init__(self, active: int) -> None:
        super().__init__([], http2=True)
        self.peer = h2.connection.H2Connection(
            config=h2.config.H2Configuration(client_side=False),
        )
        self.peer.local_settings = h2.settings.Settings(
            client=False,
            initial_values={h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: active},
        )
        self.peer.initiate_connection()
        self.requests = 0
        self.active = active
        self.paused = False
        self.zero_read_started = anyio.Event()
        self.incoming = anyio.Event()
        self.publish()

    def publish(self) -> None:
        """Expose the peer's bytes and wake the current network reader."""
        pending = self.peer.data_to_send()
        if pending:
            self._buffer.append(pending)
            self.incoming.set()
            self.incoming = anyio.Event()

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        for event in self.peer.receive_data(buffer):
            if isinstance(event, h2.events.RequestReceived):
                self.requests += 1
                self.peer.send_headers(event.stream_id, [(b":status", b"200")])
                if self.requests > self.active:
                    self.peer.send_data(event.stream_id, b"resumed", end_stream=True)
        self.publish()

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        while not self._buffer:
            if self._closed:
                return b""
            if self.paused:
                self.zero_read_started.set()
            incoming = self.incoming
            await incoming.wait()
        return await super().read(max_bytes, timeout)

    async def aclose(self) -> None:
        await super().aclose()
        self.incoming.set()

    def finish(self, stream_id: int, *, pause: bool = False) -> None:
        """Finish an admitted response, optionally stopping new stream admission."""
        if pause:
            self.paused = True
            self.peer.update_settings({h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: 0})
        self.peer.send_data(stream_id, b"complete", end_stream=True)
        self.publish()

    def resume(self) -> None:
        self.paused = False
        self.peer.update_settings({h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: 1})
        self.publish()


@pytest.mark.anyio
@pytest.mark.parametrize("active", [1, 2])
async def test_queued_request_reads_limit_increase(active: int) -> None:
    stream = ControlledPeer(active)
    origin = httpcore2.Origin(b"https", b"example.test", 443)
    done = anyio.Event()

    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as connection:
        async with AsyncExitStack() as stack:
            responses = [
                await stack.enter_async_context(connection.stream("GET", f"https://example.test/{index}"))
                for index in range(active)
            ]

            async def queued() -> None:
                response = await connection.request("GET", "https://example.test/queued")
                assert response.status == 200
                assert response.content == b"resumed"
                done.set()

            with anyio.fail_after(2):
                async with anyio.create_task_group() as group:
                    group.start_soon(queued)
                    await anyio.wait_all_tasks_blocked()
                    assert not done.is_set()

                    stream.finish(1, pause=True)
                    assert await responses[0].aread() == b"complete"
                    await responses[0].aclose()
                    await stream.zero_read_started.wait()

                    if active == 2:
                        stream.finish(3)
                        assert await responses[1].aread() == b"complete"
                        await responses[1].aclose()

                    assert not done.is_set()
                    assert stream.requests == active
                    stream.resume()
                    await done.wait()

        assert stream.requests == active + 1
        assert connection.is_idle()


@pytest.mark.anyio
@pytest.mark.parametrize("active", [1, 2])
async def test_cancelled_zero_limit_reader_releases_read_lock(active: int) -> None:
    stream = ControlledPeer(active)
    origin = httpcore2.Origin(b"https", b"example.test", 443)
    cancelled = anyio.Event()
    scope = anyio.CancelScope()

    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as connection:
        async with AsyncExitStack() as stack:
            responses = [
                await stack.enter_async_context(connection.stream("GET", f"https://example.test/{index}"))
                for index in range(active)
            ]

            async def queued() -> None:
                with scope:
                    await connection.request("GET", "https://example.test/cancelled")
                cancelled.set()

            with anyio.fail_after(2):
                async with anyio.create_task_group() as group:
                    group.start_soon(queued)
                    await anyio.wait_all_tasks_blocked()
                    stream.finish(1, pause=True)
                    assert await responses[0].aread() == b"complete"
                    await responses[0].aclose()
                    await stream.zero_read_started.wait()
                    scope.cancel()
                    await cancelled.wait()

            assert not connection.is_closed()
            assert connection.is_idle() is (active == 1)
            if active == 2:
                stream.finish(3)
                assert await responses[1].aread() == b"complete"
                await responses[1].aclose()

            assert connection.is_idle()
            stream.resume()
            with anyio.fail_after(2):
                response = await connection.request("GET", "https://example.test/after-cancellation")
            assert response.status == 200
            assert response.content == b"resumed"
            assert stream.requests == active + 1


@pytest.mark.anyio
@pytest.mark.parametrize("closure", ["local", "goaway"])
async def test_zero_limit_reader_observes_connection_closure(closure: str) -> None:
    stream = ControlledPeer(1)
    origin = httpcore2.Origin(b"https", b"example.test", 443)
    done = anyio.Event()

    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream) as connection:
        async with connection.stream("GET", "https://example.test/first") as first:

            async def queued() -> None:
                with pytest.raises((httpcore2.ConnectionNotAvailable, httpcore2.RemoteProtocolError)):
                    await connection.request("GET", "https://example.test/closed")
                done.set()

            with anyio.fail_after(2):
                async with anyio.create_task_group() as group:
                    group.start_soon(queued)
                    await anyio.wait_all_tasks_blocked()
                    stream.finish(1, pause=True)
                    assert await first.aread() == b"complete"
                    await first.aclose()
                    await stream.zero_read_started.wait()

                    if closure == "local":
                        await connection.aclose()
                    else:
                        stream.peer.close_connection()
                        stream.publish()

                    await done.wait()


@pytest.mark.anyio
async def test_cancelled_zero_reader_keeps_waiting_request_active() -> None:
    """A cancelled reader must not expire a connection another request is using."""
    stream = ControlledPeer(1)
    origin = httpcore2.Origin(b"https", b"example.test", 443)
    cancelled = anyio.CancelScope()
    first_done = anyio.Event()
    second_done = anyio.Event()

    async with httpcore2.AsyncHTTP2Connection(origin=origin, stream=stream, keepalive_expiry=0) as connection:
        async with connection.stream("GET", "https://example.test/first") as first:
            stream.finish(1, pause=True)
            assert await first.aread() == b"complete"

        async def first_reader() -> None:
            with cancelled:
                await connection.request("GET", "https://example.test/cancelled")
            first_done.set()

        async def second_reader() -> None:
            response = await connection.request("GET", "https://example.test/resumed")
            assert response.status == 200
            assert response.content == b"resumed"
            second_done.set()

        with anyio.fail_after(2):
            async with anyio.create_task_group() as group:
                group.start_soon(first_reader)
                await stream.zero_read_started.wait()
                group.start_soon(second_reader)
                await anyio.wait_all_tasks_blocked()

                cancelled.cancel()
                await first_done.wait()
                await anyio.wait_all_tasks_blocked()
                assert not connection.is_idle()
                assert not connection.has_expired()

                stream.resume()
                await second_done.wait()

        assert connection.is_idle()
        assert stream.requests == 2


class FailingPausedPeer(httpcore2.AsyncMockStream):
    """Answer one complete request, pause admission, then fail the next read."""

    def __init__(self, failure: str) -> None:
        super().__init__([], http2=True)
        self.peer = h2.connection.H2Connection(
            config=h2.config.H2Configuration(client_side=False),
        )
        self.peer.initiate_connection()
        self._buffer.append(self.peer.data_to_send())
        self.failure = failure
        self.requests = 0

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        for event in self.peer.receive_data(buffer):
            if isinstance(event, h2.events.RequestReceived):
                self.requests += 1
                self.peer.send_headers(event.stream_id, [(b":status", b"200")])
                self.peer.send_data(event.stream_id, b"complete", end_stream=True)
                self.peer.update_settings({h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: 0})
        pending = self.peer.data_to_send()
        if pending:
            self._buffer.append(pending)

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if not self._buffer and self.failure == "timeout":
            raise httpcore2.ReadTimeout()
        if not self._buffer and self.failure == "goaway":
            self.peer.close_connection(last_stream_id=1)
            self._buffer.append(self.peer.data_to_send())
        return await super().read(max_bytes, timeout)


class RecoveryBackend(httpcore2.AsyncMockBackend):
    """Return a failing connection once, then a fresh healthy connection."""

    def __init__(self, failure: str) -> None:
        super().__init__([], http2=True)
        self.first = FailingPausedPeer(failure)
        self.connects = 0

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object = None,
    ) -> httpcore2.AsyncNetworkStream:
        self.connects += 1
        if self.connects == 1:
            return self.first
        return FailingPausedPeer("eof")


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["eof", "timeout", "goaway"])
async def test_failed_admission_releases_single_pool_slot(failure: str) -> None:
    backend = RecoveryBackend(failure)
    async with httpcore2.AsyncConnectionPool(
        network_backend=backend,
        http1=False,
        http2=True,
        max_connections=1,
    ) as pool:
        first = await pool.request("GET", "http://example.test/first")
        assert first.status == 200
        assert first.content == b"complete"

        expected = httpcore2.ReadTimeout if failure == "timeout" else httpcore2.RemoteProtocolError
        with pytest.raises(expected):
            await pool.request("GET", "http://example.test/failed")
        assert pool.connections == []

        recovered = await pool.request(
            "GET",
            "http://example.test/recovered",
            extensions={"timeout": {"pool": 0.1}},
        )
        assert recovered.status == 200
        assert recovered.content == b"complete"
        assert backend.connects == 2
        assert backend.first.requests == 1
