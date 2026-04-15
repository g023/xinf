"""TurboXInf — Plugin System"""

import importlib
import os
import sys
from typing import Any, Callable, Dict, List, Optional


class PluginHook:
    """Defines a hook point in the inference pipeline."""
    PRE_TOKENIZE = "pre_tokenize"
    POST_TOKENIZE = "post_tokenize"
    PRE_GENERATE = "pre_generate"
    POST_GENERATE = "post_generate"
    PRE_SAMPLE = "pre_sample"
    POST_SAMPLE = "post_sample"
    ON_TOKEN = "on_token"
    PRE_LOAD = "pre_load"
    POST_LOAD = "post_load"


class PluginBase:
    """Base class for TurboXInf plugins."""
    name: str = "unnamed"
    version: str = "0.0.1"
    description: str = ""
    hooks: Dict[str, Callable] = {}

    def __init__(self, engine=None):
        self.engine = engine
        self.hooks = {}

    def register_hook(self, hook_name: str, callback: Callable):
        self.hooks[hook_name] = callback

    def on_load(self):
        """Called when the plugin is loaded."""
        pass

    def on_unload(self):
        """Called when the plugin is unloaded."""
        pass


class PluginManager:
    """Manages loading, registration, and execution of plugins."""

    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = plugins_dir
        self.plugins: Dict[str, PluginBase] = {}
        self._hooks: Dict[str, List[Callable]] = {}

    def load_plugin(self, plugin_path: str, engine=None) -> Optional[PluginBase]:
        """Load a plugin from a Python file."""
        if not os.path.exists(plugin_path):
            print(f"Plugin not found: {plugin_path}")
            return None

        module_name = os.path.splitext(os.path.basename(plugin_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find PluginBase subclass in module
        plugin_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and issubclass(attr, PluginBase)
                    and attr is not PluginBase):
                plugin_class = attr
                break

        if plugin_class is None:
            print(f"No PluginBase subclass found in {plugin_path}")
            return None

        plugin = plugin_class(engine=engine)
        plugin.on_load()
        self.plugins[plugin.name] = plugin

        # Register hooks
        for hook_name, callback in plugin.hooks.items():
            if hook_name not in self._hooks:
                self._hooks[hook_name] = []
            self._hooks[hook_name].append(callback)

        return plugin

    def load_all(self, engine=None):
        """Load all plugins from the plugins directory."""
        if not os.path.isdir(self.plugins_dir):
            return
        for fname in sorted(os.listdir(self.plugins_dir)):
            if fname.endswith(".py") and not fname.startswith("_"):
                self.load_plugin(os.path.join(self.plugins_dir, fname), engine)

    def run_hook(self, hook_name: str, *args, **kwargs) -> Any:
        """Execute all callbacks registered for a hook."""
        result = kwargs.get("data", args[0] if args else None)
        for callback in self._hooks.get(hook_name, []):
            ret = callback(result, **kwargs)
            if ret is not None:
                result = ret
        return result

    def unload_all(self):
        """Unload all plugins."""
        for plugin in self.plugins.values():
            plugin.on_unload()
        self.plugins.clear()
        self._hooks.clear()
