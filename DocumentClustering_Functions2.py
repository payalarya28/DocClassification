import nltk
stopwords = nltk.corpus.stopwords.words('english')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
import gensim.corpora as corpora
from gensim import corpora, models, similarities
from gensim.utils import tokenize
import spacy
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def run_lda(textlist, 
            num_topics=10,
            return_model=False,
            preprocess_docs=True):
 
   
    if preprocess_docs:
        doc_text  = [preprocess(d) for d in textlist]
        #doc_text =  [[item for sublist in doc_text for item in sublist]]
    else:
         doc_text = [d for d in textlist]   
        
   
    dictionary = corpora.Dictionary(doc_text)
   
    print('DICTIONARY')  
    print(dictionary)
    corpus = [dictionary.doc2bow(text) for text in doc_text]
        
    tfidf = models.tfidfmodel.TfidfModel(corpus)
    transformed_tfidf = tfidf[corpus]
    #print(transformed_tfidf)
    lda = models.ldamulticore.LdaMulticore(transformed_tfidf, num_topics=num_topics, id2word=dictionary)
    
    input_doc_topics = lda.get_document_topics(corpus)
    #print(input_doc_topics)
    
    return(lda, dictionary,corpus, doc_text)


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out



def run_lda_with_Bigrams(textlist, 
            num_topics=10,
            return_model=False):
 
    docwordlist =[]
    for text in textlist:
        lst =[]
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        #remove numbers
        text = re.sub(r'\d+', '', text)

        lst.append(text)
        
        text = lemmatization(lst)
        stop_words = stopwords.words('english')
        stop_words.extend(['tags','text','text_contents','contents','textcontents']) 
        vec = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words).fit(lst)
        bag_of_words = vec.transform(lst)

        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        #print(bigram_mod[doc_text[2]])

        wordlist = []
        for word, freq in words_freq:   
            print(word, freq)
            wordlist.append(word)
        
        docwordlist.append(wordlist)
    
      
    dictionary = corpora.Dictionary(docwordlist)
    print(dictionary)  
    corpus = [dictionary.doc2bow(text) for text in docwordlist]
    tfidf = models.tfidfmodel.TfidfModel(corpus)
    transformed_tfidf = tfidf[corpus]
    #print(transformed_tfidf)
    ldamodel = models.ldamulticore.LdaMulticore(transformed_tfidf, num_topics=4, id2word=dictionary)

    input_doc_topics = ldamodel.get_document_topics(corpus)
    
    return(ldamodel, dictionary,corpus, docwordlist)


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
    
    
    #stop_words = set(stopwords.words('english'))
    stop_words = stopwords.words('english')
    stop_words.extend(['''tags''','text_contents','contents','textcontents']) 
                                                      
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    result = [i for i in tokens if not i in stop_words]
    english_words = [w for w in result if (len(w)>2)]
    #english_words = [w for w in english_words if not w in stop_words]
    return(english_words)


def find_category(text, dictionary, lda):
    '''
    https://stackoverflow.com/questions/16262016/how-to-predict-the-topic-of-a-new-query-using-a-trained-lda-model-using-gensim
    
     For each query ( document in the test file) , tokenize the 
     query, create a feature vector just like how it was done while training
     and create text_corpus
    '''
    
    text_corpus = []

    for query in text:
        temp_doc = tokenize(query.strip())
        current_doc = []
        temp_doc = list(temp_doc)
        for word in range(len(temp_doc)):
            current_doc.append(temp_doc[word])

        text_corpus.append(current_doc)
    '''
     For each feature vector text, lda[doc_bow] gives the topic
     distribution, which can be sorted in descending order to print the 
     very first topic
    ''' 
    tops = []
    for text in text_corpus:
        doc_bow = dictionary.doc2bow(text)
        topics = sorted(lda[doc_bow],key=lambda x:x[1],reverse=True)[0]
        tops.append(topics)
    return(tops)


def category_label(ldamodel, topicnum):
    alltopics = ldamodel.show_topics(formatted=False)
    topic = alltopics[topicnum]
    #convert to dict
    topic = dict(topic[1])
    import operator
    return(max(topic, key=lambda key: topic[key])
          )


def read_metadata():
    df = pd.read_json('./Output/scan.json',orient='index')
        
    alltext = []
   
    for index, row in df.iterrows():      
        alltext.append((row['location'],str(row['summary'])))
        #alltext.append((row['location'],row['summary']))
    return(alltext) # Returns list of tuples [(filename, text), ... (filename,text)]
                       
                       
                       
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

