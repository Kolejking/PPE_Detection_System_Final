from pathlib import Path
from ultralytics import YOLO


class PPETracker:

    def __init__(
        self,
        model: YOLO,
        tracker_config: str,
        device: str = "cpu"
    ):
        self.model = model
        self.tracker_config = Path(tracker_config)
        self.device = device

        if not self.tracker_config.exists():
            raise FileNotFoundError(
                f"Tracker config not found: {self.tracker_config}"
            )

    def track(
        self,
        source,
        conf: float = 0.25
    ):
        return self.model.track(
            source=source,
            tracker=str(self.tracker_config),
            device=self.device,
            conf=conf,
            stream=True,
            persist=True,
            verbose=False
        ) 