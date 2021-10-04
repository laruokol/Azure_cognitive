from computer_vision.vision_base import BaseVision
from computer_vision.utils import analyze_image
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from collections import Counter


class ObjectDetector(BaseVision):
    def __init__(self, service_key: str, service_endpoint: str) -> None:
        super().__init__(service_key, service_endpoint)

    def analyze(
        self,
        image_path: str = None,
        image_dir: str = None,
        image_url: str = None,
        **kwargs
    ):

        return analyze_image(
            self.service_client,
            image_path=image_path,
            image_url=image_url,
            image_dir=image_dir,
            features=[VisualFeatureTypes.objects],
            **kwargs
        )

    @staticmethod
    def get_parents(result):
        return {
            k: [o["parent"] for o in v["objects"] if "parent" in o]
            for k, v in result.items()
        }

    @staticmethod
    def count(result, confidence: float = 0.5):
        return {
            k: Counter(
                [
                    o["object_property"]
                    for o in v["objects"]
                    if o["confidence"] > confidence
                ]
            )
            for k, v in result.items()
        }
