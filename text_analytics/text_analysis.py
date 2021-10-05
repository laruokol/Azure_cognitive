from text_analytics.text_base import BaseText
from typing import List, Union, Dict


class TextAnalyzer(BaseText):
    def __init__(
        self, service_endpoint: str, service_key: str = None, cognitive_key: str = None
    ):
        super().__init__(service_endpoint, service_key, cognitive_key)

    @staticmethod
    def _parse_docs(documents: Union[List[str], str]) -> List[str]:
        if isinstance(documents, str):
            documents = [documents]
        return documents

    @staticmethod
    def _read_text(path):
        with open(path, "r") as file:
            text = file.read()
        return text

    def detect_language(
        self,
        documents: Union[List[str], str] = None,
        document_path: str = None,
        country_hint: str = "none",
    ) -> List[Dict]:
        if document_path is not None:
            documents = self._read_text(document_path)
        documents = self._parse_docs(documents)
        try:
            response = self.service_client.detect_language(
                documents=documents, country_hint=country_hint
            )
            return [
                {
                    "language": resp.primary_language.name,
                    "confidence": resp.primary_language.confidence_score,
                }
                for resp in response
            ]
        except Exception as err:
            print("Encountered exception. {}".format(err))

    def sentiment_analysis(
        self,
        documents: Union[str, List[str]] = None,
        document_path: str = None,
    ) -> List[Dict]:
        if document_path is not None:
            documents = self._read_text(document_path)
        documents = self._parse_docs(documents)
        response = self.service_client.analyze_sentiment(documents=documents)
        result = []
        for resp in response:
            out = {
                "overall": {
                    "sentiment": resp.sentiment,
                    "scores": {
                        "positive": resp.confidence_scores.positive,
                        "neutral": resp.confidence_scores.neutral,
                        "negative": resp.confidence_scores.negative,
                    },
                },
            }
            out.update(
                {
                    "sentences": {
                        idx: {
                            "sentence": sentence.text,
                            "sentiment": sentence.sentiment,
                            "scores": {
                                "positive": sentence.confidence_scores.positive,
                                "neutral": sentence.confidence_scores.neutral,
                                "negative": sentence.confidence_scores.negative,
                            },
                        }
                        for idx, sentence in enumerate(resp.sentences, 1)
                    }
                }
            )
            result.append(out)

        return result

    def key_phrases(
        self,
        documents: Union[str, List[str]] = None,
        document_path: str = None,
    ) -> List[List[str]]:
        if document_path is not None:
            documents = self._read_text(document_path)
        documents = self._parse_docs(documents)
        response = self.service_client.extract_key_phrases(documents=documents)
        return [resp.key_phrases for resp in response]

    def entities(
        self,
        documents: Union[str, List[str]] = None,
        document_path: str = None,
    ) -> List[List[str]]:
        if document_path is not None:
            documents = self._read_text(document_path)
        documents = self._parse_docs(documents)
        response = self.service_client.recognize_entities(documents=documents)
        result = []
        for resp in response:
            result.append(
                [
                    {
                        "text": ent.text,
                        "category": ent.category,
                        "subcategory": ent.subcategory,
                        "confidence": ent.confidence_score,
                    }
                    for ent in resp.entities
                ]
            )
        return result

    def analyze(
        self,
        documents: Union[str, List[str]] = None,
        document_path: str = None,
    ) -> Dict:
        if document_path is not None:
            documents = self._read_text(document_path)
        return {
            "language": self.detect_language(documents),
            "sentiment": self.sentiment_analysis(documents),
            "key_pharsese": self.key_phrases(documents),
            "entities": self.entities(documents),
        }
