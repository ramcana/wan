from pathlib import Path

class ModelEnsurer:
    def __init__(self, *args, **kwargs):
        self.models_root = Path(kwargs.get("models_root") or "models").resolve()
    
    def ensure(self, model_id: str, variant: str | None = None):
        p = self.models_root / model_id / (variant or "default")
        p.mkdir(parents=True, exist_ok=True)
        (p / ".placeholder").write_text("stub")
        return str(p)