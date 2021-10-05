from utils import authenticate
from azure.ai.textanalytics import TextAnalyticsClient


class BaseText:
    def __init__(
        self, service_endpoint: str, service_key: str = None, cognitive_key: str = None
    ) -> None:
        credentials = authenticate(service_key, cognitive_key)
        self.service_client = TextAnalyticsClient(service_endpoint, credentials)
