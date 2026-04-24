"""
tests/test_plugins.py.
======================
Unit tests for the plugin system.
"""

import numpy as np
import pytest

from trustlens.plugins.base import BasePlugin
from trustlens.plugins.registry import PluginRegistry

# ---------------------------------------------------------------------------
# Test plugin implementations
# ---------------------------------------------------------------------------


class DummyPlugin(BasePlugin):
    name = "dummy"
    description = "A test plugin that does nothing."
    version = "0.2.0"

    def run(self, model, X, y_true, y_pred, y_prob, **kwargs):
        return {"dummy_metric": 42.0}


class InvalidPlugin:
    """Does not subclass BasePlugin."""

    name = "invalid"


class EmptyNamePlugin(BasePlugin):
    name = ""  # invalid — empty name

    def run(self, model, X, y_true, y_pred, y_prob, **kwargs):
        return {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before each test."""
    # Save original state
    original = dict(PluginRegistry._registry)
    yield
    # Restore
    PluginRegistry._registry = original


class TestPluginRegistry:
    def test_register_valid_plugin(self):
        registry = PluginRegistry()
        registry.register(DummyPlugin)
        assert "dummy" in registry

    def test_register_non_plugin_raises_type_error(self):
        registry = PluginRegistry()
        with pytest.raises(TypeError):
            registry.register(InvalidPlugin)

    def test_register_empty_name_raises_value_error(self):
        registry = PluginRegistry()
        with pytest.raises(ValueError, match="non-empty"):
            registry.register(EmptyNamePlugin)

    def test_get_returns_instance(self):
        registry = PluginRegistry()
        registry.register(DummyPlugin)
        plugin = registry.get("dummy")
        assert isinstance(plugin, DummyPlugin)

    def test_get_unknown_plugin_raises_key_error(self):
        registry = PluginRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent_plugin")

    def test_unregister(self):
        registry = PluginRegistry()
        registry.register(DummyPlugin)
        registry.unregister("dummy")
        assert "dummy" not in registry

    def test_unregister_unknown_raises(self):
        registry = PluginRegistry()
        with pytest.raises(KeyError):
            registry.unregister("ghost")

    def test_list_plugins(self):
        registry = PluginRegistry()
        registry.register(DummyPlugin)
        listing = registry.list_plugins()
        assert isinstance(listing, list)
        names = [p["name"] for p in listing]
        assert "dummy" in names

    def test_len(self):
        registry = PluginRegistry()
        initial = len(registry)
        registry.register(DummyPlugin)
        assert len(registry) == initial + 1

    def test_plugin_run_returns_dict(self):
        registry = PluginRegistry()
        registry.register(DummyPlugin)
        plugin = registry.get("dummy")
        result = plugin.run(
            model=None,
            X=np.random.randn(10, 3),
            y_true=np.array([0, 1] * 5),
            y_pred=np.array([0, 1] * 5),
            y_prob=np.random.rand(10, 2),
        )
        assert isinstance(result, dict)
        assert result["dummy_metric"] == 42.0
