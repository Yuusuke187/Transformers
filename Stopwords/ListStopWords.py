# Stop words are words that are of little or no value in Natural
# Language processing.

import nltk


def list_stop_words():
    nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words('english')
    
    print("Stop words count: ", len(stop_words))
    print("Getting the first seven stop words in the list: ", stop_words[:7])
    
    print("Returning the stopwords that only consist of one letter...")
    [sw for sw in stop_words if len(sw) == 1]
    
def identify_lemmae():
    nltk.download('wordnet')
    