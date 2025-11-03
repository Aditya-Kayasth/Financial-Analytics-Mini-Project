import nltk

print("Downloading NLTK VADER lexicon...")
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    print("VADER lexicon already downloaded.")
except LookupError:
    nltk.download('vader_lexicon')
    print("Download complete.")