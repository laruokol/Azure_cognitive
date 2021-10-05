import azure.cognitiveservices.speech as speechsdk


def translate_speech_to_text(translation_config):
    # Creates a translation recognizer using and audio file as input.
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config
    )
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