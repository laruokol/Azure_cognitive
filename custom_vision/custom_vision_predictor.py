from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from msrest.authentication import ApiKeyCredentials
from custom_vision.utils import classify_images


class CustomVisionPredictor:
    """Predictor class for Azure Custom Vision API"""

    def __init__(
        self,
        predoction_key: str,
        endpoint: str,
        project_id: str,
        published_name: str,
    ):
        self.credentials = ApiKeyCredentials(
            in_headers={"Prediction-key": predoction_key}
        )
        self.predictor = CustomVisionPredictionClient(
            endpoint=endpoint, credentials=self.credentials
        )
        self.project_id = project_id
        self.published_name = published_name

    def classify(
        self,
        image_path: str = None,
        store: bool = False,
        **kwargs,
    ):
        result = classify_images(
            project_id=self.project_id,
            predictor=self.predictor,
            published_name=self.published_name,
            image_path=image_path,
            store=store,
            **kwargs,
        )
        return result
