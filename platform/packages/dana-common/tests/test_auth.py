"""Tests for custom token engine and API key generator."""


from dana_common.auth import APIKeyGenerator, TokenEngine


class TestTokenEngine:
    def setup_method(self) -> None:
        self.engine = TokenEngine("test-secret-key-for-dana-2026")

    def test_create_and_verify_token(self) -> None:
        token = self.engine.create_token(
            user_id=1, email="test@dana.ir", tier="pro", permissions=["chat", "models"]
        )
        payload = self.engine.verify_token(token)
        assert payload is not None
        assert payload["sub"] == 1
        assert payload["email"] == "test@dana.ir"
        assert payload["tier"] == "pro"
        assert "chat" in payload["perms"]

    def test_expired_token_rejected(self) -> None:
        token = self.engine.create_token(
            user_id=1, email="test@dana.ir", tier="free", permissions=["chat"],
            expiry_minutes=-1,
        )
        payload = self.engine.verify_token(token)
        assert payload is None

    def test_tampered_token_rejected(self) -> None:
        token = self.engine.create_token(
            user_id=1, email="test@dana.ir", tier="free", permissions=["chat"]
        )
        parts = token.split(".")
        parts[1] = parts[1] + "tampered"
        tampered = ".".join(parts)
        payload = self.engine.verify_token(tampered)
        assert payload is None

    def test_wrong_secret_rejected(self) -> None:
        token = self.engine.create_token(
            user_id=1, email="test@dana.ir", tier="free", permissions=["chat"]
        )
        other_engine = TokenEngine("different-secret")
        payload = other_engine.verify_token(token)
        assert payload is None

    def test_invalid_format_rejected(self) -> None:
        assert self.engine.verify_token("not-a-token") is None
        assert self.engine.verify_token("a.b") is None
        assert self.engine.verify_token("") is None


class TestAPIKeyGenerator:
    def setup_method(self) -> None:
        self.gen = APIKeyGenerator("test-salt")

    def test_generate_key_format(self) -> None:
        key, key_hash = self.gen.generate("pro", ["chat", "models"])
        assert key.startswith("dk-p")
        assert len(key) > 20
        assert len(key_hash) == 64  # SHA-256 hex

    def test_key_hash_consistent(self) -> None:
        key, key_hash = self.gen.generate("free", ["chat"])
        assert self.gen.hash_key(key) == key_hash

    def test_unique_keys(self) -> None:
        key1, _ = self.gen.generate("free", ["chat"])
        key2, _ = self.gen.generate("free", ["chat"])
        assert key1 != key2

    def test_tier_encoding(self) -> None:
        free_key, _ = self.gen.generate("free", ["chat"])
        pro_key, _ = self.gen.generate("pro", ["chat"])
        ent_key, _ = self.gen.generate("enterprise", ["chat"])
        assert free_key.startswith("dk-f")
        assert pro_key.startswith("dk-p")
        assert ent_key.startswith("dk-e")
