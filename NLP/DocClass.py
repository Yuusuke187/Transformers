# This script will be used for Document Classification

from transformers import pipeline

# Load a pre-trained classification model from Hugging Face
# Let's use distilbert-base-uncased-finetuned-sst-2-english
classifier = pipeline("text-classification", \
    model="distilbert-base-uncased-finetuned-sst-2-english", \
    device=0)


