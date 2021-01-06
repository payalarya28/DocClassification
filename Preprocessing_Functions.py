import requests
import pandas as pd
from nltk.tokenize import word_tokenize
import regex as re
import nltk
stopwords = nltk.corpus.stopwords.words('english')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from gensim.parsing.preprocessing import remove_stopwords

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def preprocess(document):
    clean = filter_english_words(document)   
    # Remove non-english words   
    
    return(clean)

def filter_english_words(text):
    
    #remove punctuations
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    #remove numbers
    text = re.sub(r'\d+', '', text)
    
    
    stop_words = set(stopwords.words('english'))
    
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    result = [i for i in tokens if not i in ENGLISH_STOP_WORDS]
    #print (result)
    
    #english_words = word_tokenize(text)
    #english_words = [w for w in english_words if w in list(dict_words)]
    #english_words = [w for w in english_words if w not in stopwords]
    english_words = [w for w in result if (len(w)>2)]
    
    return(english_words)


def read_metadata():
    df = pd.read_json('./Output/scan.json',orient='index')
        
    alltext = []
   
    for index, row in df.iterrows():      
        alltext.append((row['title'],str(row['summary'])))
        #alltext.append((row['location'],row['summary']))
    return(alltext) # Returns list of tuples [(filename, text), ... (filename,text)]