from typing import Dict, List
import os
from azure.core.credentials import AzureKeyCredential
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient


def authenticate_client(
    service_endpoint: str, service_key: str = None, cognitive_key: str = None
):
    if service_key is not None:
        credential = AzureKeyCredential(service_key)
    elif cognitive_key is not None:
        credential = CognitiveServicesCredentials(cognitive_key)
    text_analytics_client = ComputerVisionClient(
        endpoint=service_endpoint, credential=credential
    )
    return text_analytics_client


def analyze_image(
    service_client: ComputerVisionClient,
    image_path: str = None,
    image_url: str = None,
    image_dir: str = None,
    features: List[str] = None,
    **kwargs
) -> Dict:
    if image_path is not None:
        with open(image_path, "rb") as image_stream:
            result = service_client.analyze_image_in_stream(
                image_stream, visual_features=features, **kwargs
            ).as_dict()
    elif image_url is not None:
        result = service_client.analyze_image(
            image_url, visual_features=features, **kwargs
        ).as_dict()
    elif image_dir is not None:
        files = [f for f in os.listdir(image_dir) if not f.startswith(".")]
        result = {}
        for file in files:
            with open(os.path.join(image_dir, file), "rb") as image_stream:
                result[file] = service_client.analyze_image_in_stream(
                    image_stream, visual_features=features, **kwargs
                ).as_dict()

    return result
