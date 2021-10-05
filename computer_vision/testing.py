from computer_vision.text_analysis import TextDetector
from computer_vision.image_analysis import ImageAnalyzer
from computer_vision.object_detection import ObjectDetector
from text_analytics.text_analysis import TextAnalyzer
from speech.speech_analysis import SpeechAnalyzer
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("SERVICE_KEY")
endpoint = os.getenv("ENDPOINT")

# Text Analyzer:
image_url = "https://d2jaiao3zdxbzm.cloudfront.net/wp-content/uploads/figure-65.png"
image_path = "/Users/lruokolainen/Downloads/figure-65.png"

TA = TextDetector(service_key=key, service_endpoint=endpoint)
TA.analyze(image_path=image_path)
TA.analyze(image_url=image_url)

# Image Analyzer:
image_url = "https://www.bitgab.com/uploads/1605844302-pedestrian-1605844302.jpg"
image_path = "/Users/lruokolainen/Downloads/1605844302-pedestrian-1605844302.jpg"

AV = ImageAnalyzer(service_key=key, service_endpoint=endpoint)
AV.analyze(image_path=image_path)
AV.analyze(image_url=image_url)
AV.analyze(image_url=image_url, features=["description"])
obj = AV.analyze(image_url=image_url, features=["objects"])

# Object Detector:
image_dir = "test_images"
OD = ObjectDetector(service_key=key, service_endpoint=endpoint)
result = OD.analyze(image_dir=image_dir)

OD.get_parents(result)
OD.count(result, confidence=0.5)

# Text Analysis:
document = "We went to Contoso Steakhouse located at midtown NYC last week for a dinner party, and we adore the spot! They provide marvelous food and they have a great menu. The chief cook happens to be the owner (I think his name is John Doe) and he is super nice, coming out of the kitchen and greeted us all. We enjoyed very much dining in the place! The Sirloin steak I ordered was tender and juicy, and the place was impeccably clean. You can even pre-order from their online menu at www.contososteakhouse.com, call 312-555-0176 or send email to order@contososteakhouse.com! The only complaint I have is the food didn't come fast enough. Overall I highly recommend it!"
TextAnalyzer(service_key=key, service_endpoint=endpoint).analyze(document)

# Speech:
SpeechAnalyzer(service_key=key).translate()
SpeechAnalyzer(service_key=key).transcribe()