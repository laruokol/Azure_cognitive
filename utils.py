import json
from azure.core.credentials import AzureKeyCredential
from msrest.authentication import CognitiveServicesCredentials


def load_config(path):
    with open(path, "r") as file:
        config = json.load(file)
    return config


def authenticate(service_key: str = None, cognitive_key: str = None):
    if service_key is not None:
        credential = AzureKeyCredential(service_key)
    elif cognitive_key is not None:
        credential = CognitiveServicesCredentials(cognitive_key)
    return credential