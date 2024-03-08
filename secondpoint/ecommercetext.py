import re
import string
import contractions
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer("[\w]+")
lemmatizer = WordNetLemmatizer()

def text_lower(text):
    return text.lower()

def text_remove_whitespaces(text):
    return text.strip()

def text_lemm(text):
    return " ".join([lemmatizer.lemmatize(word) for word in tokenizer.tokenize(text)])

def text_remove_stopwords(text):
    text_no_stopwords = [word for word in tokenizer.tokenize(text) if word not in stop_words]
    return " ".join(text_no_stopwords)

def text_no_ascii(text):
    text = re.sub(r'[^\x00-\x7f]', '', text)
    return text

def text_remove_punct(text):
    text_no_punct = text.translate(str.maketrans('', '', string.punctuation))
    return text_no_punct

def text_expand_contractions(text):
    return contractions.fix(text)

def text_remove_short_words(text, min_len=2):
    words = text.split()
    long_words = [word for word in words if len(word) >= min_len]
    return ' '.join(long_words)

def text_remove_long_words(text, max_len=10):
    words = text.split()
    short_words = [word for word in words if len(word) <= max_len]
    return ' '.join(short_words)

def text_remove_nonalph(text):
    return " ".join([word for word in tokenizer.tokenize(text) if word.isalpha()])

def text_preprocess(text):
    #text = text_expand_contractions(text)
    text = text_lower(text)
    text = text_remove_whitespaces(text)
    text = text_remove_punct(text)
    text = text_remove_stopwords(text)
    text = text_no_ascii(text)
    text = text_remove_nonalph(text)
    text = text_lemm(text)
    text = text_remove_short_words(text)
    text = text_remove_long_words(text)
    return text