"""Custom AES-256-GCM encryption layer for data-at-rest.

Daneshbonyan: Internal Design & Development - Custom encryption module.
NOT a standard library wrapper - implements key derivation, rotation, and versioning.
"""

from __future__ import annotations

import base64
import os
import struct

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class EncryptionEngine:
    """Custom AES-256-GCM encryption with key versioning and rotation support.

    Encrypted format: version(1 byte) + nonce(12 bytes) + ciphertext + tag(16 bytes)
    Key derivation: HKDF-SHA256 from master key + context
    """

    VERSION = 1

    def __init__(self, master_key: str) -> None:
        self._master_key = bytes.fromhex(master_key)
        self._keys: dict[int, bytes] = {}
        self._current_version = 1
        self._derive_key(1)

    def _derive_key(self, version: int) -> bytes:
        """Derive versioned encryption key using HKDF."""
        context = f"dana-encryption-v{version}".encode()
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=struct.pack(">I", version),
            info=context,
        )
        key = hkdf.derive(self._master_key)
        self._keys[version] = key
        return key

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext string. Returns base64-encoded ciphertext."""
        key = self._keys.get(self._current_version)
        if key is None:
            key = self._derive_key(self._current_version)
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        # Format: version(1) + nonce(12) + ciphertext+tag
        packed = struct.pack("B", self._current_version) + nonce + ciphertext
        return base64.urlsafe_b64encode(packed).decode("ascii")

    def decrypt(self, encrypted: str) -> str:
        """Decrypt base64-encoded ciphertext. Supports key versioning."""
        raw = base64.urlsafe_b64decode(encrypted)
        version = struct.unpack("B", raw[:1])[0]
        nonce = raw[1:13]
        ciphertext = raw[13:]
        key = self._keys.get(version)
        if key is None:
            key = self._derive_key(version)
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")

    def rotate_key(self) -> int:
        """Rotate to a new key version. Old versions still supported for decryption."""
        self._current_version += 1
        self._derive_key(self._current_version)
        return self._current_version


class FieldEncryptor:
    """Encrypts/decrypts individual database fields.

    Usage:
        enc = FieldEncryptor(master_key)
        encrypted_email = enc.encrypt_field("user@example.com")
        email = enc.decrypt_field(encrypted_email)
    """

    def __init__(self, master_key: str) -> None:
        self._engine = EncryptionEngine(master_key)

    def encrypt_field(self, value: str) -> str:
        return self._engine.encrypt(value)

    def decrypt_field(self, encrypted: str) -> str:
        return self._engine.decrypt(encrypted)
