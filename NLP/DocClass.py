# This script will be used for Document Classification

from transformers import pipeline

def classify_text(document_text):
    # Load a pre-trained classification model from Hugging Face
    # Let's use distilbert-base-uncased-finetuned-sst-2-english
    classifier = pipeline("text-classification", \
        model="distilbert-base-uncased-finetuned-sst-2-english", \
        device=0)

    # document_text = """
    # Raiden is a vertically scrolling shooting game (shmup) released by Seibu Kaihatsu in 1990.
    # """
    if document_text == "":
        document_text = input("Enter the text that will be clasified: ")

    # Classify the text
    classification = classifier(document_text)

    # Print out the classification
    print(classification)
