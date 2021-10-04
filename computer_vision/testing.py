from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from computer_vision.text_analysis import TextAnalyzer
from computer_vision.image_analysis import ImageAnalyzer
from computer_vision.object_detection import ObjectDetector
from collections import Counter

creds = CognitiveServicesCredentials(key)
service_client = ComputerVisionClient(endpoint, creds)

# Text Analyzer:
image_url = "https://d2jaiao3zdxbzm.cloudfront.net/wp-content/uploads/figure-65.png"
image_path = "/Users/lruokolainen/Downloads/figure-65.png"

TV = TextVision(service_key=key, service_endpoint=endpoint)
TV.analyze(image_path=image_path)
TV.analyze(image_url=image_url)

# Image Analyzer:
image_url = "https://www.bitgab.com/uploads/1605844302-pedestrian-1605844302.jpg"
image_path = "/Users/lruokolainen/Downloads/1605844302-pedestrian-1605844302.jpg"

AV = AnalyzeVision(service_key=key, service_endpoint=endpoint)
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
