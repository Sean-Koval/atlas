import re
from bs4 import BeautifulSoup
from nltk import pos_tag
from nltk.corpus import stopwords
import nltk
from fastapi import HTTPException
from pydantic import BaseModel, validator


#nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
"""
def extract_article_metadata(article_html):
    soup = BeautifulSoup(article_html, 'html.parser')
    title = soup.find('h3').text
    article_text = soup.find('article').text
    tagged_tokens = pos_tag(re.findall(r'\b[A-Za-z][A-Za-z-]*\b', article_text))
    proper_nouns = []
    for i, (word, pos) in enumerate(tagged_tokens):
        if pos == 'NNP' and word not in stop_words:
            if word.lower() == 'ventures' or (i > 0 and tagged_tokens[i-1][1] == 'NNP'):
                proper_nouns.append(word)
            else:
                acronym_regex = r'^[A-Z]+$'
                if re.match(acronym_regex, word):
                    if i < len(tagged_tokens) - 1 and tagged_tokens[i+1][1] == 'NNS':
                        proper_nouns.append(word)
    return {'title': title, 'venture_capital_firms': proper_nouns}
"""

"""
import re
import string

from nltk import word_tokenize, pos_tag, download
from nltk.corpus import wordnet, stopwords

download('stopwords')
download('averaged_perceptron_tagger')

# Create a list of stop words to exclude from the model
stop_words = set(stopwords.words('english') + list(string.punctuation))

def extract_venture_capital_firms(text):
    # Tokenize the text and tag the tokens with their part of speech
    tokens = pos_tag(word_tokenize(text))
    
    # Initialize a list to store the venture capital firms
    venture_capital_firms = []
    
    # Iterate through the tokens and check for potential venture capital firms
    for i, (token, pos) in enumerate(tokens):
        # Ignore stop words and tokens that are not proper nouns
        if token in stop_words or pos not in ['NNP', 'NNPS']:
            continue
            
        # Check if the token is a known venture capital firm
        if wordnet.synsets(token, pos='n') and 'venture.n.01' in [synset.name() for synset in wordnet.synsets(token, pos='n')]:
            venture_capital_firms.append(token)
        # Check if the token is the beginning of a multi-word venture capital firm
        elif i+1 < len(tokens) and tokens[i+1][1] == 'NNP':
            venture_capital_firms.append(token + ' ' + tokens[i+1][0])
            
    return venture_capital_firms

text = "Acme Ventures and Founders VC are two well-known venture capital firms. Sequia Capital is an up and coming player in the VC space."
venture_capital_firms = extract_venture_capital_firms(text)
print(venture_capital_firms)

"""


class Article(BaseModel):
    name: str
    date: str
    text: str

#    @validator('__model__')
#    def check_required_keys(cls, values):
#        if not all(key in values for key in ('name', 'date', 'text')):
#            raise ValueError("Invalid article format: missing required keys")
#        return values

class ProcessedArticle(BaseModel):
    name: str
    text: str
    date: str
    vc_firms: list[str]


import re
import multiprocessing

# Create a list of known venture capital firms
venture_capital_firms = ['Acme Ventures', 'Founders VC', 'Sequia Capital']

# Compile the regular expressions for the venture capital firms
patterns = [re.compile(venture_capital_firm) for venture_capital_firm in venture_capital_firms]

def extract_venture_capital_firms(text, patterns):
    # Initialize a list to store the venture capital firms
    extracted_firms = []
    
    # Iterate through the patterns
    for pattern in patterns:
        # Find all occurrences of the venture capital firm in the text
        matches = pattern.findall(text)
        
        # Add the matches to the list of extracted firms
        extracted_firms += matches
        
    return extracted_firms
"""
def process_article(article):
    # Ensure that article is a dictionary
    if not isinstance(article, dict):
        raise TypeError('Expected a dictionary, got {}'.format(type(article)))
    
    # Extract the text and venture capital firms from the article
    text = article['text']
    extracted_firms = extract_venture_capital_firms(text, patterns)
    
    # Return a dictionary with the name, text, date, and venture capital firms
    return {
        'name': article['name'],
        'text': text,
        'date': article['date'],
        'vc_firms': extracted_firms
    }
    """
# not sure if I should return object here or in the FASTAPI endpoint
def process_article(article): # -> ProcessedArticle:
    try:
        # If article is a list, process each element in the list
        if isinstance(article, list):
            return [process_article(a) for a in article]
    
        # Otherwise, assume that article is a dictionary
        text = article['text']
        extracted_firms = extract_venture_capital_firms(text, patterns)
    
        #return ProcessedArticle(        
        #    name=article.name,
        #    text=text,
        #    date=article.date,
        #    vc_firms=extracted_firms
        #)
        # Return a dictionary with the name, text, date, and venture capital firms
        return {
            'name': article['name'],
            'text': text,
            'date': article['date'],
            'vc_firms': extracted_firms
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Error processing article: {}".format(str(e))
        )
"""
def process_articles(articles):
    # Ensure that articles is a list of dictionaries
    if not all(isinstance(article, dict) for article in articles):
        raise TypeError('Expected a list of dictionaries, got {}'.format(articles))
    
    # Use multiprocessing to parallelize the processing of the articles
    with multiprocessing.Pool() as pool:
        processed_articles = pool.map(process_article, articles)
        
    return processed_articles

"""

import concurrent.futures

def process_articles(articles):
    # Create a thread pool with 4 threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit the process_article() function as a task to be executed by the thread pool
        processed_articles = list(executor.map(process_article, articles))
        
    return processed_articles

@app.post("/process_articles")
def process_articles_endpoint(articles: list[Article]):
    try:
        processed_articles = process_articles(articles)
        return processed_articles
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Error processing articles: {}".format(str(e))
        )

# Define a list of articles
articles = [
    {
        'name': 'Article 1',
        'date': '2022-01-01',
        'text': 'Acme Ventures and Founders VC are two well-known venture capital firms. Sequia Capital is an up and coming player in the VC space.'
    },
    {
        'name': 'Article 2',
        'date': '2022-01-02',
        'text': 'Acme Ventures is a leading venture capital firm in the tech'
    }
]




output = process_articles(articles)

print(output)



"""
import re
import string

import spacy

nlp = spacy.load("en_core_web_sm")


# Create a list of stop words to exclude from the model
stop_words = set(nlp.Defaults.stop_words + list(string.punctuation))

def extract_venture_capital_firms(text):
    # Tokenize and tag the text using the spacy library
    doc = nlp(text)
    
    # Initialize a list to store the venture capital firms
    venture_capital_firms = []
    
    # Iterate through the tokens and check for potential venture capital firms
    for i, token in enumerate(doc):
        # Ignore stop words and tokens that are not proper nouns
        if token.text in stop_words or token.pos_ != "PROPN":
            continue
            
        # Check if the token is a known venture capital firm
        if token.text.endswith("Capital") and "venture" in token.text.lower():
            venture_capital_firms.append(token.text)
        # Check if the token is the beginning of a multi-word venture capital firm
        elif i+1 < len(doc) and doc[i+1].pos_ == "PROPN" and doc[i+1].text.endswith("Capital"):
            venture_capital_firms.append(token.text + ' ' + doc[i+1].text)
            
    return venture_capital_firms

text = "Acme Ventures and Founders VC are two well-known venture capital"
"""


# Output: {'title': 'Article Title', 'venture_capital_firms': ['acme ventures', 'Founders VC', 'Sequia Capital']}

"""
Use a different part-of-speech tagger: The pos_tag function from the nltk library uses the averaged perceptron tagger, which may not always produce the most accurate POS tags. You could try using a different POS tagger, such as the Stanford POS tagger or the Spacy library, to see if it performs better on your data.

Use a named entity recognition (NER) model: NER models are specifically designed to identify and classify named entities, such as people, organizations, and locations, in text. You could use a pre-trained NER model, such as Spacy's en_core_web_sm model, to extract venture capital firms from the text.

Use a machine learning classifier: You could train a machine learning classifier, such as a support vector machine (SVM) or a random forest classifier, on a labeled dataset of venture capital firms. This would allow you to use more advanced features, such as word embeddings or character-level features, to improve the model's performance.

Use a regular expression to extract venture capital firms: You could use a regular expression to extract venture capital firms from the text. For example, you could use a pattern like '\b[A-Z][a-z]*(?: [A-Z][a-z]*)? Capital\b' to match multi-word venture capital firms that consist of one or more capitalized words followed by "Capital".

Use a combination of these approaches: You could also try combining these approaches to see if it improves the model's performance. For example, you could use a POS tagger to identify proper nouns, and then use a regular expression or a machine learning classifier to filter out false positives.
"""