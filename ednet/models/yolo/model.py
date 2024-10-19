from pathlib import Path

from ednet.engine.model import Model
from ednet.models import yolo
from ednet.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel
from ednet.utils import ROOT, yaml_load
import os

class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        path = Path(model)
        super().__init__(model=model, task=task, verbose=verbose)
        
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
        }

class EDNet(YOLO):
    """EdgeDroneNet model (Default EDNet-tiny) based on YOLOv10."""

    def __init__(self, model=None, task=None, verbose=False):
        """Initialize CGTA_YOLO model with a fixed model path."""
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        fixed_model_path = current_dir / "../../cfg/models/ednet/ednet-t.yaml"
        super().__init__(model=model or fixed_model_path, task=task, verbose=verbose)

