import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import json
import uuid
import requests
from typing import List, Union, Dict


def read_text(path):
    with open(path, "r") as file:
        text = file.read()
    return text


def load_config(path):
    with open(path, "r") as file:
        config = json.load(file)
    return config


def authenticate_client(service_key, service_endpoint):
    text_analytics_client = TextAnalyticsClient(
        endpoint=service_endpoint, credential=AzureKeyCredential(service_key)
    )
    return text_analytics_client


def from_mic(speech_config):
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    print("Speak into your microphone:")
    result = speech_recognizer.recognize_once_async().get()
    print(result.text)
    return result


def translate(translation_config):
    speech_recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config
    )

    print("Speak into your microphone:")
    result = speech_recognizer.recognize_once_async().get()
    print(result.text)
    return result


def translate_speech_to_text(translation_config):
    # Creates a translation recognizer using and audio file as input.
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config
    )
    print("Say something...")
    result = recognizer.recognize_once_async().get()

    # Check the result
    if result.reason == speechsdk.ResultReason.TranslatedSpeech:
        print(
            "RECOGNIZED '{}': {}".format(
                translation_config.speech_recognition_language, result.text
            )
        )
        toLanguage = translation_config.target_languages[0]
        print(
            "TRANSLATED into {}: {}".format(toLanguage, result.translations[toLanguage])
        )
    elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("RECOGNIZED: {} (text could not be translated)".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print(
            "NOMATCH: Speech could not be recognized: {}".format(
                result.no_match_details
            )
        )
    elif result.reason == speechsdk.ResultReason.Canceled:
        print("CANCELED: Reason={}".format(result.cancellation_details.reason))
        if result.cancellation_details.reason == speechsdk.CancellationReason.Error:
            print(
                "CANCELED: ErrorDetails={}".format(
                    result.cancellation_details.error_details
                )
            )

    return result


def actk_sentiment_analysis(
    client: TextAnalyticsClient,
    texts: Union[str, List[str]],
):
    if isinstance(texts, str):
        texts = [texts]
    response = client.analyze_sentiment(documents=texts)[0]
    print("Document Sentiment: {}".format(response.sentiment))
    out = {
        "overall": {
            "sentiment": response.sentiment,
            "scores": {
                "positive": response.confidence_scores.positive,
                "neutral": response.confidence_scores.neutral,
                "negative": response.confidence_scores.negative,
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
                for idx, sentence in enumerate(response.sentences, 1)
            }
        }
    )

    return out


def actk_translate(
    texts: Union[str, List[str]],
    service_key: str,
    location: str = "northeurope",
    fromLanguage: str = "fi",
    toLanguage: Union[str, List[str]] = "en",
    api_version: str = "3.0",
):
    params = {"api-version": api_version, "from": fromLanguage, "to": toLanguage}
    headers = {
        "Ocp-Apim-Subscription-Key": service_key,
        "Ocp-Apim-Subscription-Region": location,
        "Content-type": "application/json",
        "X-ClientTraceId": str(uuid.uuid4()),
    }
    if isinstance(texts, str):
        texts = [texts]
    body = [{"text": t} for t in texts]

    endpoint = "https://api.cognitive.microsofttranslator.com"
    path = "/translate"
    constructed_url = endpoint + path
    request = requests.post(
        constructed_url,
        params=params,
        headers=headers,
        json=body,
    )

    return request.json()


def actk_detect_language(
    client: TextAnalyticsClient,
    documents: Union[List[str], str],
):
    if isinstance(documents, str):
        documents = [documents]
    try:
        response = client.detect_language(documents=documents, country_hint="none")[0]
        return response
    except Exception as err:
        print("Encountered exception. {}".format(err))
