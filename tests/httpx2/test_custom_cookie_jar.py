"""Custom cookie jars can update state through their response hooks."""

from http.cookiejar import CookieJar

import pytest
from typing_extensions import override

import httpx2
from httpx2._cookies import IndexedCookieJar


def install_session_cookie(jar: CookieJar) -> None:
    """Set the cookie the authenticated endpoint requires."""
    cookies = httpx2.Cookies()
    cookies.set("session", "token", domain="example.com")
    for cookie in cookies.jar:
        jar.set_cookie(cookie)


class SessionJar(CookieJar):
    """Install a session cookie through the supported standard-jar hook."""

    @override
    def extract_cookies(self, response: object, request: object) -> None:
        install_session_cookie(self)


class IndexedSessionJar(IndexedCookieJar):
    """Keep the response hook of an indexed-jar subclass."""

    @override
    def extract_cookies(self, response: object, request: object) -> None:
        install_session_cookie(self)


def session_endpoint(request: httpx2.Request) -> httpx2.Response:
    """Require the cookie installed after the first response."""
    if request.url.path == "/login":
        return httpx2.Response(200)
    return httpx2.Response(200 if request.headers.get("cookie") == "session=token" else 401)


@pytest.mark.parametrize("jar_type", [SessionJar, IndexedSessionJar])
def test_custom_cookie_jar_observes_a_response_without_cookie_headers(
    jar_type: type[CookieJar],
) -> None:
    transport = httpx2.MockTransport(session_endpoint)
    with httpx2.Client(transport=transport, cookies=jar_type()) as client:
        assert client.get("https://example.com/login").status_code == 200
        assert client.get("https://example.com/private").status_code == 200


@pytest.mark.anyio
@pytest.mark.parametrize("jar_type", [SessionJar, IndexedSessionJar])
async def test_async_custom_cookie_jar_observes_a_response_without_cookie_headers(
    jar_type: type[CookieJar],
) -> None:
    transport = httpx2.MockTransport(session_endpoint)
    async with httpx2.AsyncClient(transport=transport, cookies=jar_type()) as client:
        assert (await client.get("https://example.com/login")).status_code == 200
        assert (await client.get("https://example.com/private")).status_code == 200
