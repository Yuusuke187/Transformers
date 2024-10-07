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


    