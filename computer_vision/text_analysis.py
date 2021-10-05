from computer_vision.vision_base import BaseVision
from typing import Dict
import os


class TextDetector(BaseVision):
    """
    Help.
    """

    def __init__(self, service_key: str, service_endpoint: str) -> None:
        super().__init__(service_key, service_endpoint)
        self._key = service_key

    def analyze(
        self,
        image_path: str = None,
        image_url: str = None,
        image_dir: str = None,
        **kwargs
    ) -> Dict:
        if image_path is not None:
            with open(image_path, "rb") as image_stream:
                result = self.service_client.recognize_printed_text_in_stream(
                    image_stream, **kwargs
                )
        elif image_url is not None:
            result = self.service_client.recognize_printed_text(image_url, **kwargs)
        elif image_dir is not None:
            files = os.listdir(image_dir)
            for file in files:
                with open(os.path.join(image_dir, file), "rb") as image_stream:
                    result = self.service_client.recognize_printed_text_in_stream(
                        image_stream, **kwargs
                    )

        return result.as_dict()
