# This script will be used for Document Summarization

from transformers import pipeline

# Load a pre-trained summarization model from Facebook
summarizer = pipeline("summarization", \
    model="facebook/bart-large-cnn")

# Create a long string to go in a document
long_text = """
AI has shown remarkable progress since the early days of the internet.
It has revolutionized the way we interact with information.
It has made it easy to create and share data.
It has changed the way we think about healthcare.
It has made healthcare more accessible.
It has changed the way we live.
(This was all generated with Codeium)
"""

# Summarize the text
summary = summarizer(long_text, max_length=100, min_length=30, \
    do_sample=False)

# Print out the summary

