import requests, uuid
from typing import List, Union


class TextTranslator:
    def __init__(
        self,
        service_key: str,
        service_location: str = "northeurope",
        api_version: str = "3.0",
    ):
        self.service_key = service_key
        self.service_location = service_location
        self.api_version = api_version
        self.endpoint = "https://api.cognitive.microsofttranslator.com/translate"
        self.headers = {
            "Ocp-Apim-Subscription-Key": service_key,
            "Ocp-Apim-Subscription-Region": service_location,
            "Content-type": "application/json",
            "X-ClientTraceId": str(uuid.uuid4()),
        }

    def translate(
        self,
        text: Union[str, List[str]],
        from_lang: str = "en",
        to_lang: Union[str, List[str]] = "fi",
    ):
        if isinstance(text, str):
            text = [text]
        body = [{"text": t} for t in text]
        params = {"api-version": self.api_version, "from": from_lang, "to": to_lang}
        request = requests.post(
            self.endpoint, params=params, headers=self.headers, json=body
        )
        return request.json()