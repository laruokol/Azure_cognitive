from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from utils import authenticate
import pandas as pd


class BaseVision:
    def __init__(
        self, service_endpoint: str, service_key: str = None, cognitive_key: str = None
    ) -> None:

        self.vision_key = service_key
        self.url = service_endpoint
        credentials = authenticate(service_key, cognitive_key)
        self.service_client = ComputerVisionClient(self.url, credentials)

    def get_metadata(self, response):
        """
        Collects metadata for each image
        in the response object to pandas.data.frame
        """

        tmp = {k: pd.DataFrame(v["metadata"], index=[0]) for k, v in response.items()}
        for k, v in tmp.items():
            tmp[k]["image"] = k
            tmp[k]["request_id"] = v["request_id"]

        metadata = pd.concat(tmp.values(), ignore_index=True, sort=False)

        return metadata
