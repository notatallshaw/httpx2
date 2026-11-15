"""Keep HTTP/2 responses independent while sharing one network reader."""

import math
from types import TracebackType

import anyio
import h2.config
import h2.connection
import h2.events
import pytest
from typing_extensions import override

import httpcore2
from httpcore2._async.http2 import AsyncHTTP2Connection
from httpcore2._synchronization import AsyncLock


class ResponseGateStream(httpcore2.AsyncMockStream):
    """Serve actual HTTP/2 responses only when the test releases each stream."""

    def __init__(self) -> None:
        super().__init__([])
        self.server = h2.connection.H2Connection(config=h2.config.H2Configuration(client_side=False))
        self.server.initiate_connection()
        self.sender, self.receiver = anyio.create_memory_object_stream[bytes | Exception](math.inf)
        self.sender.send_nowait(self.server.data_to_send())
        self.requests = {stream_id: anyio.Event() for stream_id in (1, 3, 5)}
        self.active_reads = 0
        self.max_active_reads = 0
        self.read_count = 0
        self.read_timeouts: list[float | None] = []

    @override
    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self.read_count += 1
        self.read_timeouts.append(timeout)
        self.active_reads += 1
        self.max_active_reads = max(self.max_active_reads, self.active_reads)
        try:
            data = await self.receiver.receive()
            if isinstance(data, Exception):
                raise data
            return data
        finally:
            self.active_reads -= 1

    @override
    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        for event in self.server.receive_data(buffer):
            if isinstance(event, h2.events.RequestReceived):
                self.requests[event.stream_id].set()
        data = self.server.data_to_send()
        if data:
            self.sender.send_nowait(data)

    def respond(self, stream_id: int, body: bytes = b"") -> None:
        """Release one complete response without releasing other streams."""
        self.server.send_headers(stream_id, [(b":status", b"200")], end_stream=not body)
        if body:
            self.server.send_data(stream_id, body, end_stream=True)
        self.sender.send_nowait(self.server.data_to_send())

    @override
    async def aclose(self) -> None:
        self.sender.close()
        self.receiver.close()
        await super().aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("reserved_turn", [True, False])
async def test_buffered_response_does_not_wait_for_another_stream(
    reserved_turn: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = ResponseGateStream()
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    completed: set[int] = set()

    async with AsyncHTTP2Connection(origin=origin, stream=stream) as connection:
        monkeypatch.setattr(connection._read_lock, "reserves_waiter_turn", reserved_turn)

        async def request(stream_id: int) -> None:
            response = await connection.request("GET", f"https://example.com/{stream_id}")
            assert response.status == 200
            assert response.content == b""
            completed.add(stream_id)

        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                for stream_id in (1, 3, 5):
                    group.start_soon(request, stream_id)
                    await stream.requests[stream_id].wait()
                    await anyio.wait_all_tasks_blocked()

                # Stream1 reads stream5's response while stream3 still needs I/O.
                stream.respond(5)
                await anyio.wait_all_tasks_blocked()
                assert completed == {5}
                assert stream.max_active_reads == 1

                stream.respond(3)
                await anyio.wait_all_tasks_blocked()
                assert completed == {3, 5}

                stream.respond(1)

    assert completed == {1, 3, 5}
    assert stream.max_active_reads == 1


@pytest.mark.anyio
async def test_mixed_read_timeouts_keep_reader_rotation() -> None:
    stream = ResponseGateStream()
    origin = httpcore2.Origin(b"https", b"example.com", 443)

    async with AsyncHTTP2Connection(origin=origin, stream=stream) as connection:

        async def request(stream_id: int, read_timeout: float) -> None:
            response = await connection.request(
                "GET", f"https://example.com/{stream_id}", extensions={"timeout": {"read": read_timeout}}
            )
            assert response.status == 200

        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                for stream_id, read_timeout in ((1, 100.0), (3, 0.01), (5, 100.0)):
                    group.start_soon(request, stream_id, read_timeout)
                    await stream.requests[stream_id].wait()
                    await anyio.wait_all_tasks_blocked()

                stream.server.ping(b"12345678")
                stream.sender.send_nowait(stream.server.data_to_send())
                await anyio.wait_all_tasks_blocked()
                assert stream.read_timeouts[-1] == 0.01

                for stream_id in (1, 3, 5):
                    stream.respond(stream_id)

    assert stream.max_active_reads == 1


@pytest.mark.anyio
@pytest.mark.parametrize("condition", ["response", "flow", "connection"])
async def test_ready_stream_or_connection_progress(condition: str) -> None:
    stream = ResponseGateStream()
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    request = httpcore2.Request("GET", "https://example.com/", headers={b"host": b"example.com"})

    async with AsyncHTTP2Connection(origin=origin, stream=stream) as connection:

        async def consume() -> None:
            response = await connection.handle_async_request(request)
            reads = stream.read_count
            # A thread can arrive here after another reader made its stream ready.
            if condition == "flow":
                await connection._receive_events(request, flow_stream_id=1)
            elif condition == "response":
                await connection._receive_events(request, stream_id=1)
            else:
                stream.server.ping(b"12345678")
                stream.sender.send_nowait(stream.server.data_to_send())
                await connection._receive_events(request)
            assert stream.read_count == reads + (condition == "connection")
            assert await response.aread() == b"payload"
            await response.aclose()

        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                group.start_soon(consume)
                await stream.requests[1].wait()
                stream.respond(1, b"payload")


@pytest.mark.anyio
@pytest.mark.parametrize("cancelled_stream", [1, 3])
async def test_cancelling_reader_or_waiter_keeps_other_streams_live(cancelled_stream: int) -> None:
    stream = ResponseGateStream()
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    completed: set[int] = set()
    scopes: dict[int, anyio.CancelScope] = {}

    async with AsyncHTTP2Connection(origin=origin, stream=stream) as connection:

        async def request(stream_id: int) -> None:
            with anyio.CancelScope() as scope:
                scopes[stream_id] = scope
                response = await connection.request("GET", f"https://example.com/{stream_id}")
                assert response.status == 200
                completed.add(stream_id)

        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                for stream_id in (1, 3, 5):
                    group.start_soon(request, stream_id)
                    await stream.requests[stream_id].wait()
                    await anyio.wait_all_tasks_blocked()

                scopes[cancelled_stream].cancel()
                await anyio.wait_all_tasks_blocked()
                stream.respond(5)
                await anyio.wait_all_tasks_blocked()
                assert completed == {5}

                stream.respond(3 if cancelled_stream == 1 else 1)

    assert completed == {1, 3, 5} - {cancelled_stream}
    assert stream.max_active_reads == 1


@pytest.mark.anyio
@pytest.mark.parametrize("cancelled_stream", [3, 5])
@pytest.mark.parametrize("reserved_turn", [True, False])
async def test_cancellation_during_buffered_peer_handoff(
    cancelled_stream: int, reserved_turn: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = ResponseGateStream()
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    scopes: dict[int, anyio.CancelScope] = {}
    completed: set[int] = set()
    tasks: dict[int, int] = {}
    cancelled = False

    async with AsyncHTTP2Connection(origin=origin, stream=stream) as connection:
        monkeypatch.setattr(connection._read_lock, "reserves_waiter_turn", reserved_turn)
        original_exit = AsyncLock.__aexit__

        async def cancel_before_release(
            lock: AsyncLock,
            exc_type: type[BaseException] | None = None,
            exc_value: BaseException | None = None,
            traceback: TracebackType | None = None,
        ) -> None:
            nonlocal cancelled
            if lock is connection._read_lock and anyio.get_current_task().id == tasks.get(3) and not cancelled:
                cancelled = True
                scopes[cancelled_stream].cancel()
            await original_exit(lock, exc_type, exc_value, traceback)

        monkeypatch.setattr(AsyncLock, "__aexit__", cancel_before_release)

        async def request(stream_id: int) -> None:
            tasks[stream_id] = anyio.get_current_task().id
            with anyio.CancelScope() as scope:
                scopes[stream_id] = scope
                response = await connection.request("GET", f"https://example.com/{stream_id}")
                assert response.status == 200
                completed.add(stream_id)

        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                for stream_id in (1, 3, 5):
                    group.start_soon(request, stream_id)
                    await stream.requests[stream_id].wait()
                    await anyio.wait_all_tasks_blocked()

                stream.respond(5)
                await anyio.wait_all_tasks_blocked()
                assert completed == ({5} if cancelled_stream == 3 else set())

                for stream_id in (1, 3):
                    if stream_id != cancelled_stream:
                        stream.respond(stream_id)
                await anyio.wait_all_tasks_blocked()
                assert completed == {1, 3, 5} - {cancelled_stream}

    assert stream.max_active_reads == 1


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["read_error", "goaway"])
async def test_connection_failure_wakes_every_waiting_stream(failure: str) -> None:
    stream = ResponseGateStream()
    origin = httpcore2.Origin(b"https", b"example.com", 443)
    errors: list[Exception] = []
    error = httpcore2.ReadError("connection lost")

    async with AsyncHTTP2Connection(origin=origin, stream=stream) as connection:

        async def request(stream_id: int) -> None:
            try:
                await connection.request("GET", f"https://example.com/{stream_id}")
            except (httpcore2.ReadError, httpcore2.RemoteProtocolError) as exc:
                errors.append(exc)

        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                for stream_id in (1, 3, 5):
                    group.start_soon(request, stream_id)
                    await stream.requests[stream_id].wait()
                    await anyio.wait_all_tasks_blocked()

                if failure == "read_error":
                    stream.sender.send_nowait(error)
                else:
                    # These streams were processed, so they cannot be retried.
                    stream.server.close_connection(last_stream_id=5)
                    stream.sender.send_nowait(stream.server.data_to_send())

    assert len(errors) == 3
    if failure == "read_error":
        assert all(exc is error for exc in errors)
    else:
        assert all(isinstance(exc, httpcore2.RemoteProtocolError) for exc in errors)
    assert stream.max_active_reads == 1
