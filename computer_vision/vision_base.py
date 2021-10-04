from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import pandas as pd


class BaseVision:
    def __init__(self, service_key: str, service_endpoint: str) -> None:

        self.vision_key = service_key
        self.url = service_endpoint
        self.service_client = ComputerVisionClient(
            self.url, CognitiveServicesCredentials(self.vision_key)
        )

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
