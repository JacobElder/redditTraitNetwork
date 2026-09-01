from rtn.rater import MockRater
from rtn.rater.cache import CallCache, make_key
from rtn.traits import load_traits


def test_cache_roundtrip_and_key_sensitivity(tmp_path):
    cache = CallCache(tmp_path / "c.sqlite")
    a = make_key({"account_hash": "h", "prompt_version": "p1", "task": "e2_pair",
                  "trait": "honest->kind", "replicate": 0, "config_hash": "abc"})
    b = make_key({"account_hash": "h", "prompt_version": "p1", "task": "e2_pair",
                  "trait": "honest->kind", "replicate": 0, "config_hash": "xyz"})
    assert a != b
    cache.put(a, parts={"task": "e2_pair"}, request="q", response='{"rating": 5}',
              model="mock", usage={})
    assert cache.get(a)["response"] == '{"rating": 5}'
    assert cache.get(b) is None


def test_mock_rater_free_mode_is_deterministic():
    v = load_traits("traits/traits.yaml")
    r1 = MockRater(vocab=v, seed=3)
    r2 = MockRater(vocab=v, seed=3)
    parts = {"task": "e2_pair", "trait": "honest->kind", "replicate": 1}
    assert r1.complete("p", call_parts=parts).text == r2.complete("p", call_parts=parts).text
