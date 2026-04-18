"""Unit tests for config_loader module."""

import argparse
import pytest
from config_loader import load_config, merge_config_with_args, validate_config


class TestLoadConfig:
    """load_config: parse YAML files, reject bad input."""

    def test_valid_yaml_loaded(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("contract: MNQ\nsize: 2\n")
        result = load_config(str(cfg))
        assert result["contract"] == "MNQ"
        assert result["size"] == 2

    def test_yml_extension_accepted(self, tmp_path):
        cfg = tmp_path / "config.yml"
        cfg.write_text("contract: NQ\n")
        result = load_config(str(cfg))
        assert result["contract"] == "NQ"

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("")
        result = load_config(str(cfg))
        assert result == {}

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "missing.yaml"))

    def test_unsupported_extension_raises(self, tmp_path):
        cfg = tmp_path / "config.json"
        cfg.write_text("{}")
        with pytest.raises(ValueError, match="Only YAML"):
            load_config(str(cfg))

    def test_nested_values_preserved(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("trading:\n  risk_amount: 100\n  size: 2\n")
        result = load_config(str(cfg))
        assert result["trading"]["risk_amount"] == 100


class TestMergeConfigWithArgs:
    """merge_config_with_args: CLI args override config, None args do not."""

    def _args(self, **kwargs):
        ns = argparse.Namespace(**kwargs)
        return ns

    def test_args_override_config(self):
        config = {"contract": "MNQ", "size": 1}
        args = self._args(contract="NQ", size=None)
        merged = merge_config_with_args(config, args)
        assert merged["contract"] == "NQ"

    def test_none_args_do_not_override(self):
        config = {"size": 3}
        args = self._args(size=None)
        merged = merge_config_with_args(config, args)
        assert merged["size"] == 3

    def test_new_keys_from_args_added(self):
        config = {"contract": "MNQ"}
        args = self._args(risk_amount=100)
        merged = merge_config_with_args(config, args)
        assert merged["risk_amount"] == 100
        assert merged["contract"] == "MNQ"

    def test_config_not_mutated(self):
        config = {"contract": "MNQ", "size": 1}
        args = self._args(contract="NQ")
        merge_config_with_args(config, args)
        assert config["contract"] == "MNQ"

    def test_empty_config_uses_args(self):
        args = self._args(contract="ES", size=2)
        merged = merge_config_with_args({}, args)
        assert merged["contract"] == "ES"
        assert merged["size"] == 2


class TestValidateConfig:
    """validate_config: raise ValueError for missing required fields."""

    def _full_config(self):
        return {
            "account": "ACC123",
            "contract": "MNQ",
            "username": "user",
            "apikey": "key",
            "strategy": "cisd_ote",
            "model": "model.onnx",
            "scaler": "scaler.pkl",
            "market_hub": "https://hub",
            "base_url": "https://api",
        }

    def test_valid_config_passes(self):
        validate_config(self._full_config())  # should not raise

    def test_missing_one_field_raises(self):
        cfg = self._full_config()
        del cfg["apikey"]
        with pytest.raises(ValueError, match="apikey"):
            validate_config(cfg)

    def test_none_value_treated_as_missing(self):
        cfg = self._full_config()
        cfg["account"] = None
        with pytest.raises(ValueError, match="account"):
            validate_config(cfg)

    def test_all_missing_raises_all_names(self):
        with pytest.raises(ValueError):
            validate_config({})

    def test_extra_keys_ignored(self):
        cfg = self._full_config()
        cfg["extra_key"] = "some_value"
        validate_config(cfg)  # should not raise
