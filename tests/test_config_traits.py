from rtn.config import load_config
from rtn.traits import load_traits


def test_config_hash_changes_with_prompt_version(tmp_path):
    cfg = load_config("config/default.yaml")
    assert len(cfg.hash8) == 8
    ov = tmp_path / "ov.yaml"
    ov.write_text("prompts:\n  version: p9\n")
    cfg2 = load_config("config/default.yaml", str(ov))
    assert cfg.hash8 != cfg2.hash8


def test_config_hash_ignores_cache_path(tmp_path):
    cfg = load_config("config/default.yaml")
    ov = tmp_path / "ov.yaml"
    ov.write_text("cache:\n  path: somewhere/else.sqlite\n")
    assert load_config("config/default.yaml", str(ov)).hash8 == cfg.hash8


def test_trait_vocab_valence_balanced_and_paired():
    v = load_traits("traits/traits.yaml")
    assert v.k == 40
    assert sum(x == "pos" for x in v.valence.values()) == 20
    for a, b in v.antonym.items():
        assert v.antonym[b] == a
