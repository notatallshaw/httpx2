import logging

import httpcore2
from httpcore2._trace import Trace


class UnusableKwargs(dict[str, object]):
    def __bool__(self) -> bool:
        raise AssertionError("Disabled tracing inspected its context")


def test_disabled_trace_skips_unused_context() -> None:
    logger = logging.Logger("httpcore2.test", level=logging.INFO)
    request = httpcore2.Request("GET", "https://example.test/")
    marker = object()
    trace = Trace("request", logger, request, UnusableKwargs())

    assert trace.return_value is None
    with trace:
        trace.return_value = marker

    assert trace.return_value is marker


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
    def callback(_name: str, _info: dict[str, object]) -> None:
        pass

    logger = logging.Logger("httpcore2.test", level=logging.INFO)
    request = httpcore2.Request("GET", "https://example.test/", extensions={"trace": callback})
    first = Trace("request", logger, request)
    second = Trace("request", logger, request)
    first.return_value = object()

    assert first is not second
    assert first.should_trace
    assert second.return_value is None
