"""Tests for custom encryption engine."""

from dana_common.encryption import EncryptionEngine, FieldEncryptor


class TestEncryptionEngine:
    def setup_method(self) -> None:
        self.engine = EncryptionEngine("0123456789abcdef0123456789abcdef")

    def test_encrypt_decrypt_roundtrip(self) -> None:
        plaintext = "Hello, Dana AI Platform!"
        encrypted = self.engine.encrypt(plaintext)
        decrypted = self.engine.decrypt(encrypted)
        assert decrypted == plaintext

    def test_different_ciphertexts(self) -> None:
        plaintext = "same input"
        e1 = self.engine.encrypt(plaintext)
        e2 = self.engine.encrypt(plaintext)
        assert e1 != e2  # Different nonces

    def test_unicode_support(self) -> None:
        plaintext = "سلام دانا - هوش مصنوعی ایرانی"
        encrypted = self.engine.encrypt(plaintext)
        decrypted = self.engine.decrypt(encrypted)
        assert decrypted == plaintext

    def test_key_rotation(self) -> None:
        plaintext = "secret data"
        encrypted_v1 = self.engine.encrypt(plaintext)
        new_version = self.engine.rotate_key()
        assert new_version == 2
        encrypted_v2 = self.engine.encrypt(plaintext)
        # Both versions should decrypt
        assert self.engine.decrypt(encrypted_v1) == plaintext
        assert self.engine.decrypt(encrypted_v2) == plaintext

    def test_empty_string(self) -> None:
        encrypted = self.engine.encrypt("")
        assert self.engine.decrypt(encrypted) == ""


class TestFieldEncryptor:
    def test_field_roundtrip(self) -> None:
        enc = FieldEncryptor("0123456789abcdef0123456789abcdef")
        original = "user@example.com"
        encrypted = enc.encrypt_field(original)
        assert encrypted != original
        assert enc.decrypt_field(encrypted) == original
