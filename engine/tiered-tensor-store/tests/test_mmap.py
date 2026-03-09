"""Tests for MmapPool and SSDStore."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tiered_tensor_store.mmap_pool import MmapPool
from tiered_tensor_store.ssd_direct import SSDStore


# ---------------------------------------------------------------------------
# MmapPool
# ---------------------------------------------------------------------------

@pytest.fixture
def mmap_pool(tmp_path):
    return MmapPool(base_dir=str(tmp_path / "mmap"))


def test_mmap_write_and_read(mmap_pool):
    t = torch.randn(8, 8)
    mmap_pool.write("t1", t)
    loaded = mmap_pool.to_tensor("t1")
    assert torch.allclose(t, loaded, atol=1e-5)


def test_mmap_allocate_shape(mmap_pool):
    mm = mmap_pool.allocate("a1", shape=(4, 4), dtype=np.float32)
    assert mm.shape == (4, 4)
    assert mm.dtype == np.float32


def test_mmap_allocate_zero_filled(mmap_pool):
    mm = mmap_pool.allocate("a2", shape=(3, 3), fill=0.0)
    assert (mm == 0).all()


def test_mmap_exists(mmap_pool):
    t = torch.ones(2, 2)
    assert not mmap_pool.exists("new_key")
    mmap_pool.write("new_key", t)
    assert mmap_pool.exists("new_key")


def test_mmap_free_removes_file(mmap_pool, tmp_path):
    t = torch.ones(2, 2)
    mmap_pool.write("del_me", t)
    assert mmap_pool.exists("del_me")
    mmap_pool.free("del_me")
    assert not mmap_pool.exists("del_me")


def test_mmap_keys_list(mmap_pool):
    mmap_pool.write("k1", torch.ones(2))
    mmap_pool.write("k2", torch.ones(3))
    assert set(mmap_pool.keys()) == {"k1", "k2"}


def test_mmap_total_bytes(mmap_pool):
    t = torch.randn(16, 16)  # 1024 floats * 4 bytes = 4096 bytes
    mmap_pool.write("big", t)
    assert mmap_pool.total_bytes() >= t.element_size() * t.numel()


def test_mmap_get_unknown_raises(mmap_pool):
    with pytest.raises(KeyError):
        mmap_pool.get("does_not_exist")


# ---------------------------------------------------------------------------
# SSDStore
# ---------------------------------------------------------------------------

@pytest.fixture
def ssd(tmp_path):
    return SSDStore(base_dir=str(tmp_path / "ssd"))


def test_ssd_save_and_load(ssd):
    t = torch.randn(4, 4)
    ssd.save("t1", t)
    loaded = ssd.load("t1")
    assert torch.allclose(t, loaded, atol=1e-5)


def test_ssd_exists(ssd):
    t = torch.ones(2)
    assert not ssd.exists("k")
    ssd.save("k", t)
    assert ssd.exists("k")


def test_ssd_delete(ssd):
    t = torch.ones(2)
    ssd.save("del_me", t)
    ssd.delete("del_me")
    assert not ssd.exists("del_me")


def test_ssd_load_missing_raises(ssd):
    with pytest.raises(FileNotFoundError):
        ssd.load("never_saved")


def test_ssd_list_keys(ssd):
    ssd.save("a", torch.ones(1))
    ssd.save("b", torch.ones(2))
    assert set(ssd.list_keys()) == {"a", "b"}
