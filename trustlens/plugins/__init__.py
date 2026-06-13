"""
trustlens.plugins.
==================
Plugin system for TrustLens.

The plugin system lets external contributors extend TrustLens without
modifying the core library. A plugin is a Python class that:

1. Inherits from ``BasePlugin``.
2. Implements the ``run()`` method.
3. Is registered via ``PluginRegistry.register()``.

TrustLens ships with a minimal core; domain-specific features
(e.g., medical imaging fairness, NLP toxicity probes) are plugins.

Quick plugin authoring
----------------------
>>> from trustlens.plugins.base import BasePlugin
>>> from trustlens.plugins.registry import PluginRegistry
>>>
>>> class MyPlugin(BasePlugin):
...   name = "my_plugin"
...   description = "Computes a custom metric."
...
...   def run(self, model, X, y_true, y_pred, y_prob, **kwargs):
...     # ... your logic here ...
...     return {"custom_metric": 42.0}
>>>
>>> PluginRegistry().register(MyPlugin)
"""

from trustlens.plugins.base import BasePlugin
from trustlens.plugins.registry import PluginRegistry

__all__ = ["BasePlugin", "PluginRegistry"]
