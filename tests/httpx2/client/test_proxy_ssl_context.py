from __future__ import annotations

import asyncio
import platform
import ssl
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import trustme
import truststore

import httpcore2
import httpx2


@pytest.fixture
def created_contexts(monkeypatch: pytest.MonkeyPatch) -> list[ssl.SSLContext]:
    """Record real default contexts without replacing their root-loading behavior."""
    contexts: list[ssl.SSLContext] = []
    original = httpcore2.default_ssl_context

    def factory(*, _trust_env: bool = True) -> ssl.SSLContext:
        context = original(_trust_env=_trust_env)
        contexts.append(context)
        return context

    monkeypatch.setattr(httpcore2, "default_ssl_context", factory)
    return contexts


@pytest.fixture(autouse=True)
def proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://localhost:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:8080")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(sys, "platform", "linux")


@contextmanager
def closing(client: httpx2.Client | httpx2.AsyncClient) -> Iterator[httpx2.Client | httpx2.AsyncClient]:
    """Close either public client after constructor-only checks."""
    try:
        yield client
    finally:
        if isinstance(client, httpx2.AsyncClient):
            asyncio.run(client.aclose())
        else:
            client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_proxy_pools_load_default_ca_file_once(
    asynchronous: bool,
    monkeypatch: pytest.MonkeyPatch,
    created_contexts: list[ssl.SSLContext],
    cert_authority: trustme.CA,
    tmp_path: Path,
) -> None:
    ca_file = tmp_path / "ca.pem"
    cert_authority.cert_pem.write_to_path(ca_file)
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_file))
    client_type = httpx2.AsyncClient if asynchronous else httpx2.Client

    with closing(client_type()) as client:
        assert len(created_contexts) == 1
        context = created_contexts[0]
        assert context.get_ca_certs(binary_form=True) == [ssl.PEM_cert_to_DER_cert(ca_file.read_text())]
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname
        # Generated target pools use the same settings; proxy TLS has its own context.
        for transport in (client._transport, *client._mounts.values()):
            if transport is not None:
                assert isinstance(transport, (httpx2.HTTPTransport, httpx2.AsyncHTTPTransport))
                assert transport._pool._ssl_context is context


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("system", ["Windows", "Darwin"])
def test_proxy_pools_keep_separate_platform_contexts(
    asynchronous: bool,
    system: str,
    monkeypatch: pytest.MonkeyPatch,
    created_contexts: list[ssl.SSLContext],
) -> None:
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(sys, "platform", "win32" if system == "Windows" else "darwin")
    client_type = httpx2.AsyncClient if asynchronous else httpx2.Client
    with closing(client_type()):
        assert len(created_contexts) == 3
        assert len({id(context) for context in created_contexts}) == 3


@pytest.mark.parametrize("asynchronous", [False, True])
def test_custom_transport_does_not_prepare_an_unused_context(
    asynchronous: bool,
    monkeypatch: pytest.MonkeyPatch,
    created_contexts: list[ssl.SSLContext],
) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", "/missing/ca.pem")
    transport = httpx2.MockTransport(lambda request: httpx2.Response(200))
    client_type = httpx2.AsyncClient if asynchronous else httpx2.Client
    with closing(client_type(transport=transport)):
        assert not created_contexts


@pytest.mark.parametrize("asynchronous", [False, True])
def test_client_certificate_configuration_keeps_separate_contexts(
    asynchronous: bool,
    cert_pem_file: str,
    cert_private_key_file: str,
    created_contexts: list[ssl.SSLContext],
) -> None:
    client_type = httpx2.AsyncClient if asynchronous else httpx2.Client
    with pytest.warns(DeprecationWarning), closing(client_type(cert=(cert_pem_file, cert_private_key_file))):
        assert len(created_contexts) == 3


@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_proxy_ignores_ca_environment_when_requested(
    asynchronous: bool,
    monkeypatch: pytest.MonkeyPatch,
    created_contexts: list[ssl.SSLContext],
) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", "/missing/ca.pem")
    client_type = httpx2.AsyncClient if asynchronous else httpx2.Client
    with closing(client_type(proxy="http://localhost:8080", trust_env=False)):
        assert len(created_contexts) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("explicit_context", [False, True])
def test_explicit_verification_settings_do_not_load_default_roots(
    asynchronous: bool,
    explicit_context: bool,
    created_contexts: list[ssl.SSLContext],
) -> None:
    verify = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT) if explicit_context else False
    client_type = httpx2.AsyncClient if asynchronous else httpx2.Client
    with closing(client_type(verify=verify)):
        assert not created_contexts


@pytest.mark.parametrize("asynchronous", [False, True])
def test_injected_truststore_contexts_remain_separate(
    asynchronous: bool,
    created_contexts: list[ssl.SSLContext],
) -> None:
    client_type = httpx2.AsyncClient if asynchronous else httpx2.Client
    truststore.inject_into_ssl()
    try:
        with closing(client_type()):
            assert len(created_contexts) == 3
            assert len({id(context) for context in created_contexts}) == 3
    finally:
        truststore.extract_from_ssl()
