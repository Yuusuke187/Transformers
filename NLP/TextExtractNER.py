# Try running this in an isolated environment
# If you created an environment called my_env, then run
# .\my_env\Scripts\activate.bat (in Windows)

from transformers import pipeline

ner_pipeline = pipeline("ner", \
    model="dbmdz/bert-large-cased-finetuned-conll03-english")

text = """
The Simpsons premiered on December 17, 1989, and it is continuing to this day.
"""

# Perform a Name Entity Recognition task
entities = ner_pipeline(text)

# Show the extracted entities
for entity in entities:
    print(f"Entity: {entity['word']}, \
        Label: {entity['entity']}")

# This should be it for now.

    