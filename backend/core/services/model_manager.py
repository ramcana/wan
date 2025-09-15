class ModelManager:
    """Minimal stub to satisfy imports; extend if your code actually uses it."""
    def __init__(self):
        self._registry = {}

    def register(self, name: str, pipeline_cls):
        self._registry[name] = pipeline_cls

    def get(self, name: str):
        return self._registry[name]