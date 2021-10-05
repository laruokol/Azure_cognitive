from computer_vision.vision_base import BaseVision
from computer_vision.utils import analyze_image
from typing import Dict, List
import os
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes


class ImageAnalyzer(BaseVision):
    """Analyze images with Azure Computer Vision API"""

    def __init__(self, service_key: str, service_endpoint: str) -> None:
        super().__init__(service_key, service_endpoint)
        self._key = service_key

        self.visual_features = {
            "tags": VisualFeatureTypes.tags,
            "color": VisualFeatureTypes.color,
            "categories": VisualFeatureTypes.categories,
            "description": VisualFeatureTypes.description,
            "faces": VisualFeatureTypes.faces,
            "objects": VisualFeatureTypes.objects,
            "brads": VisualFeatureTypes.brands,
            "adult": VisualFeatureTypes.adult,
            "image_type": VisualFeatureTypes.image_type,
        }

    def analyze(
        self,
        image_path: str = None,
        image_url: str = None,
        image_dir: str = None,
        features: List[str] = None,
        **kwargs
    ) -> Dict:
        if features is None:
            features = [self.visual_features.get("tags")]
        elif isinstance(features, str):
            features = [features]
            features = [self.visual_features.get(f) for f in features]

        features = [f for f in features if f in self.visual_features.keys()]
        assert len(features) > 0, "No valid visual features provided"

        return analyze_image(
            self.service_client,
            image_path=image_path,
            image_url=image_url,
            image_dir=image_dir,
            features=features,
            **kwargs
        )
