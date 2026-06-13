"""
trustlens.plugins.registry.
============================
Plugin registry — a singleton that tracks all registered TrustLens plugins.

Design
------
The registry implements a simple singleton pattern backed by a class-level
dictionary. Plugins are registered either:

1. Explicitly via ``PluginRegistry().register(MyPlugin)``
2. Automatically via ``entry_points`` (future: ``trustlens.plugins`` group)

This design allows third-party packages to contribute plugins without
modifying the TrustLens source.
"""

from __future__ import annotations

import logging

from trustlens.plugins.base import BasePlugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Singleton registry for TrustLens plugins.

    All registered plugins are stored in the class-level ``_registry``
    dict so that they persist across instantiations.

    Examples
    --------
    >>> registry = PluginRegistry()
    >>> registry.register(MyPlugin)
    >>> plugin = registry.get("my_plugin")
    >>> result = plugin.run(model, X, y_true, y_pred, y_prob)
    """

    _registry: dict[str, type[BasePlugin]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, plugin_class: type[BasePlugin]) -> None:
        """
        Register a plugin class.

        Parameters
        ----------
        plugin_class : Type[BasePlugin]
          A class (not instance) that subclasses ``BasePlugin``.

        Raises
        ------
        TypeError
          If ``plugin_class`` does not subclass ``BasePlugin``.
        ValueError
          If ``plugin_class.name`` is empty or already registered.
        """
        if not issubclass(plugin_class, BasePlugin):
            raise TypeError(f"{plugin_class} must subclass trustlens.plugins.base.BasePlugin.")
        if not plugin_class.name:
            raise ValueError("Plugin class must define a non-empty 'name' class attribute.")
        if plugin_class.name in self._registry:
            logger.warning("Plugin '%s' is already registered. Overwriting.", plugin_class.name)

        self._registry[plugin_class.name] = plugin_class
        logger.info("Registered plugin: '%s'", plugin_class.name)

    def unregister(self, name: str) -> None:
        """Remove a plugin from the registry by name."""
        if name not in self._registry:
            raise KeyError(f"No plugin named '{name}' is registered.")
        del self._registry[name]
        logger.info("Unregistered plugin: '%s'", name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> BasePlugin:
        """
        Retrieve and instantiate a registered plugin by name.

        Parameters
        ----------
        name : str
          Plugin identifier (must match ``plugin_class.name``).

        Returns
        -------
        BasePlugin
          A fresh instance of the requested plugin.

        Raises
        ------
        KeyError
          If no plugin with that name is registered.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "(none)"
            raise KeyError(f"Plugin '{name}' is not registered. Available plugins: [{available}]")
        plugin_class = self._registry[name]
        instance = plugin_class()

        if not instance.validate():
            logger.warning("Plugin '%s' failed validation — skipping.", name)

        return instance

    def list_plugins(self) -> list[dict[str, str]]:
        """
        Return a list of all registered plugin summaries.

        Returns
        -------
        list[dict]
          Each dict contains ``name``, ``description``, and ``version``.
        """
        return [
            {
                "name": cls.name,
                "description": cls.description,
                "version": cls.version,
            }
            for cls in self._registry.values()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return (
            f"PluginRegistry(n_plugins={len(self._registry)}, "
            f"plugins={list(self._registry.keys())})"
        )
