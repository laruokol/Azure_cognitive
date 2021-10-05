import azure.cognitiveservices.speech as speechsdk
from speech.utils import translate_speech_to_text


class SpeechAnalyzer:
    """Speech analysis using Azure Speech API"""

    def __init__(self, service_key: str, service_location: str = "northeurope"):
        self.config = speechsdk.translation.SpeechTranslationConfig(
            subscription=service_key, region=service_location
        )

    def translate(self, from_lang: str = "fi-FI", to_lang="en"):
        self.config.speech_recognition_language = from_lang
        self.config.add_target_language(to_lang)
        print("Speak into your microphone:")
        return translate_speech_to_text(self.config)

    def transcribe(self):
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.config)
        print("Speak into your microphone:")
        result = speech_recognizer.recognize_once_async().get()
        print(result.text)
        return result