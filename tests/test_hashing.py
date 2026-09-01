import pytest

from rtn.ingest.hashing import UserMap, hash_username


def test_hash_is_stable_and_salted():
    assert hash_username("Spez", salt="s1") == hash_username("  spez ", salt="s1")
    assert hash_username("spez", salt="s1") != hash_username("spez", salt="s2")
    assert len(hash_username("spez", salt="s1")) == 16


def test_usermap_appends(tmp_path):
    um = UserMap(tmp_path / "usermap.json")  # gitignore check covers repo 'secrets/'
    h = um.add("SomeUser", salt="s1")
    assert um.get("someuser") == h
    um2 = UserMap(tmp_path / "usermap.json")
    assert um2.get("someuser") == h


def test_hash_requires_salt(monkeypatch):
    monkeypatch.delenv("RTN_HASH_SALT", raising=False)
    with pytest.raises(RuntimeError):
        hash_username("spez")
