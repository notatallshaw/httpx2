"""Unbounded event waits retain cancellation without allocating deadlines."""

from contextlib import AbstractContextManager

import anyio
import pytest
import trio

from httpcore2 import PoolTimeout
from httpcore2._synchronization import AsyncEvent


@pytest.mark.anyio
async def test_unbounded_wait_does_not_construct_a_deadline(
    anyio_backend_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    deadlines: list[float | None] = []
    if anyio_backend_name == "trio":
        original_trio = trio.fail_after

        def count_trio(delay: float) -> AbstractContextManager[trio.CancelScope]:
            deadlines.append(delay)
            return original_trio(delay)

        monkeypatch.setattr(trio, "fail_after", count_trio)
    else:
        original_anyio = anyio.fail_after

        def count_anyio(delay: float | None) -> AbstractContextManager[anyio.CancelScope]:
            deadlines.append(delay)
            return original_anyio(delay)

        monkeypatch.setattr(anyio, "fail_after", count_anyio)

    event = AsyncEvent()
    event.set()
    await event.wait()
    assert deadlines == []

    await event.wait(1)
    assert deadlines == [1]


@pytest.mark.anyio
async def test_unbounded_wait_keeps_outer_cancellation() -> None:
    event = AsyncEvent()
    finished = anyio.Event()

    async def wait() -> None:
        try:
            with pytest.raises(anyio.get_cancelled_exc_class()):
                await event.wait()
        finally:
            finished.set()

    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(wait)
            await anyio.wait_all_tasks_blocked()
            assert not finished.is_set()
            group.cancel_scope.cancel()

    assert finished.is_set()


@pytest.mark.anyio
async def test_finite_wait_still_reports_pool_timeout() -> None:
    event = AsyncEvent()
    with pytest.raises(PoolTimeout):
        await event.wait(0)


@pytest.mark.anyio
async def test_outer_deadline_keeps_its_own_error() -> None:
    event = AsyncEvent()
    with pytest.raises(TimeoutError), anyio.fail_after(0):
        await event.wait()
