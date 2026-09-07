from pathlib import Path
from ultralytics import YOLO


class PPEDetector:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def predict(self, source, conf: float = 0.25):
        return self.model.predict(
            source=source,
            conf=conf,
            device=self.device,
            verbose=False
        )
    