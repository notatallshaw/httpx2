import logging

import httpcore2
from httpcore2._trace import Trace


class CountingKwargs(dict[str, object]):
    """Record context inspection in enabled and disabled trace controls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def __bool__(self) -> bool:
        self.calls += 1
        return False


def test_disabled_trace_skips_unused_context() -> None:
    logger = logging.Logger("httpcore2.test", level=logging.INFO)
    request = httpcore2.Request("GET", "https://example.test/")
    marker = object()
    kwargs = CountingKwargs()
    trace = Trace("request", logger, request, kwargs)

    assert kwargs.calls == 0
    assert trace.return_value is None
    with trace:
        trace.return_value = marker

    assert trace.return_value is marker
    assert kwargs.calls == 0


def test_trace_subclass_keeps_instance_state() -> None:
    class CustomTrace(Trace):
        pass

    logger = logging.Logger("httpcore2.test", level=logging.INFO)
    request = httpcore2.Request("GET", "https://example.test/")
    kwargs = {"key": "value"}
    trace = CustomTrace("request", logger, request, kwargs)

    assert not trace.should_trace
    assert trace.name == "request"
    assert trace.logger is logger
    assert trace.kwargs is kwargs
    assert trace.prefix == "test"


def test_enabled_trace_keeps_per_request_state() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    def callback(name: str, info: dict[str, object]) -> None:
        events.append((name, info))

    logger = logging.Logger("httpcore2.test", level=logging.INFO)
    request = httpcore2.Request("GET", "https://example.test/", extensions={"trace": callback})
    kwargs = CountingKwargs()
    first = Trace("request", logger, request, kwargs)
    second = Trace("request", logger, request)

    assert first is not second
    assert first.should_trace
    assert second.return_value is None
    assert kwargs.calls == 1

    marker = object()
    with first:
        first.return_value = marker
    with second:
        assert second.return_value is None

    assert events == [
        ("test.request.started", {}),
        ("test.request.complete", {"return_value": marker}),
        ("test.request.started", {}),
        ("test.request.complete", {"return_value": None}),
    ]
