from __future__ import annotations

import os
import platform
import ssl
from pathlib import Path

import pytest
import trustme

import httpcore2
from httpcore2 import _ssl as ssl_config


def _client_handshake(context: ssl.SSLContext, certificate: trustme.LeafCert) -> None:
    """Complete real TLS verification without a network listener."""
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    certificate.configure_cert(server_context)
    client_incoming, client_outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
    server_incoming, server_outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
    client = context.wrap_bio(client_incoming, client_outgoing, server_hostname="localhost")
    server = server_context.wrap_bio(server_incoming, server_outgoing, server_side=True)

    while True:
        try:
            client.do_handshake()
        except ssl.SSLWantReadError:
            server_incoming.write(client_outgoing.read())
            try:
                server.do_handshake()
            except ssl.SSLWantReadError:
                pass
            client_incoming.write(server_outgoing.read())
        else:
            server_incoming.write(client_outgoing.read())
            server.do_handshake()
            return


def _context_with_paths(
    monkeypatch: pytest.MonkeyPatch,
    ca_directory: Path,
    fallback_file: Path,
    compiled_file: Path,
) -> ssl.SSLContext:
    """Point the public factory at controlled compiled roots and its fallback."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    paths = ssl.get_default_verify_paths()._replace(
        openssl_cafile=str(compiled_file),
        openssl_capath=str(ca_directory),
    )
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: paths)
    monkeypatch.setattr(ssl_config, "CA_FILE_CANDIDATES", [str(fallback_file)])
    return httpcore2.default_ssl_context()


@pytest.mark.parametrize("suffix", ["\N{ARABIC-INDIC DIGIT ZERO}", "\N{FULLWIDTH DIGIT ZERO}"])
def test_unicode_hash_suffix_does_not_suppress_ca_file_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    fallback_authority = trustme.CA()
    fallback_file = tmp_path / "fallback.pem"
    fallback_authority.cert_pem.write_to_path(fallback_file)
    ca_directory = tmp_path / "certs"
    ca_directory.mkdir()
    fallback_authority.cert_pem.write_to_path(ca_directory / ("01234567." + suffix))

    context = _context_with_paths(monkeypatch, ca_directory, fallback_file, tmp_path / "missing.pem")
    _client_handshake(context, fallback_authority.issue_cert("localhost"))


def test_ascii_hash_directory_keeps_its_trust_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority = trustme.CA(organization_name="HTTPX2 tests", organization_unit_name="hash-directory")
    ca_directory = tmp_path / "certs"
    ca_directory.mkdir()
    # OpenSSL subject hash for the fixed organization and unit above.
    authority.cert_pem.write_to_path(ca_directory / "9d5e43b6.0")
    unrelated = trustme.CA()
    fallback_file = tmp_path / "fallback.pem"
    unrelated.cert_pem.write_to_path(fallback_file)

    original = os.path.isfile

    def isfile(path: str) -> bool:
        assert Path(path) != ca_directory / "9d5e43b6.0"
        return original(path)

    monkeypatch.setattr(os.path, "isfile", isfile)
    context = _context_with_paths(monkeypatch, ca_directory, fallback_file, tmp_path / "missing.pem")
    _client_handshake(context, authority.issue_cert("localhost"))

    with pytest.raises(ssl.SSLCertVerificationError):
        _client_handshake(context, unrelated.issue_cert("localhost"))


def test_compiled_ca_file_remains_usable_with_unicode_hash_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = trustme.CA()
    compiled_file = tmp_path / "compiled.pem"
    authority.cert_pem.write_to_path(compiled_file)
    ca_directory = tmp_path / "certs"
    ca_directory.mkdir()
    (ca_directory / "01234567.\N{ARABIC-INDIC DIGIT ZERO}").write_text("unused")

    context = _context_with_paths(monkeypatch, ca_directory, tmp_path / "missing-fallback.pem", compiled_file)
    _client_handshake(context, authority.issue_cert("localhost"))


@pytest.mark.parametrize("layout", ["uppercase", "lowercase-alias", "casefold-lookup", "valid-cafile"])
def test_uppercase_hash_entry_uses_actual_lowercase_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    authority = trustme.CA(organization_name="HTTPX2 tests", organization_unit_name="hash-directory")
    ca_directory = tmp_path / "certs"
    ca_directory.mkdir()
    lowercase = ca_directory / "9d5e43b6.0"
    uppercase = lowercase.with_name(lowercase.name.upper())
    assert uppercase.name != lowercase.name
    authority.cert_pem.write_to_path(uppercase)
    fallback = authority if layout in ("uppercase", "valid-cafile") else trustme.CA()
    fallback_file = tmp_path / "fallback.pem"
    fallback.cert_pem.write_to_path(fallback_file)

    if layout in ("lowercase-alias", "casefold-lookup") and not lowercase.is_file():
        lowercase.hardlink_to(uppercase)
    if layout == "casefold-lookup":
        original = os.listdir

        def entries(path: str) -> list[str]:
            # Model a case-insensitive listing; the lowercase alias and TLS are real.
            return [uppercase.name] if Path(path) == ca_directory else original(path)

        monkeypatch.setattr(os, "listdir", entries)

    compiled_file = fallback_file if layout == "valid-cafile" else tmp_path / "missing.pem"
    context = _context_with_paths(monkeypatch, ca_directory, fallback_file, compiled_file)
    _client_handshake(context, authority.issue_cert("localhost"))
