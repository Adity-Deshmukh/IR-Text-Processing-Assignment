# -*- coding: utf-8 -*-
"""
# Information Retrieval Assignment 
# Topic : Text Processing
# Group Number : 11

We start by importing the required python libraries and modules.
"""
import math
import sys
import pickle
import string
import os
from datetime import datetime, timedelta
import numpy as np

from bs4 import BeautifulSoup as bsoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import regexp_tokenize

def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{} hrs, {} mins, {} secs".format(hours, minutes, seconds)

# This cell contains various auxilary functions used throughout the notebook
# used during query processing

def getTermFreq(freq):
    '''
        int -> double

        Input : term_frequecny
        Output : normalized term frequency

        Description : 
        returns the log to the base 10 noramlized value 
    '''

    return 1 + math.log10(freq)

"""
The buildIndex function uses the wikipedia pages and extracts the documents present in each file. 

We are using beautiful soup to extract the documents enclosed within doc tag and removing all the other HTML tags present within in each document.

For the preprocessing we will not be using any kind of stemming or lemmatization or normalization. We are only removing the punctuations between words for example "Alpha - Beta" will be converted to "Alpha Beta" however "Alpha-Beta" will remain the same.

Finally we store in our index the document ids, document titles and the posting list for the terms.
"""

def buildIndex(corpuspath):
    '''
        path -> dict

        Input : corpuspath 
        Output : {'num_docs','posting_list','norm_factor','doc_titles','doc_ids'}

        Description : 
        returns the inverted index generated from the wikipedia corpus 
        (AA folder) along with document ids, document titles and their 
        normalization constants

    '''
    startTime = datetime.now()

    postings = {}
    num_docs = 0
    norm_factors = []
    doc_titles = []
    doc_ids = []

    i_doc = 0

    for foldername in os.listdir(corpuspath):
        # Constructing index from the 'AA' folder only
        if foldername != 'AA':
            continue
        for filename in os.listdir(os.path.join(corpuspath, foldername)):
            filepath = os.path.join(corpuspath, foldername, filename)
            print("\tProcessing " + str(filepath))
            # Opening the file in read only mode
            with open(filepath,'r',encoding='utf-8') as file:
                soup = file.read()
            soup = bsoup(soup, "lxml")
            # Extracting all the documents
            docs = soup.find_all('doc')

            num_docs += len(docs)
            # Getting all the doc_titles and doc_ids
            doc_titles += [doc.get('title') for doc in docs]
            doc_ids += [doc.get('id') for doc in docs]

            for doc in docs:
                # Removing punctuations from the documents and tokenizing
                doctxt = doc.get_text()
                doctxt = doctxt.translate(str.maketrans('','',string.punctuation))
                tokens = word_tokenize(doctxt)
                tokens = [word.lower() for word in tokens]

                # Getting the term frequency for the term in the document
                doc_dict = {}
                for token in tokens:
                    if token in doc_dict:
                        doc_dict[token]=doc_dict[token]+1
                    else:
                        doc_dict[token]=1
                # Calculating the normalization factor and add the doc id and 
                # term frequency to the index
                norm_factors.append(0)
                for key in doc_dict:
                    if key in postings:
                        postings[key].append((i_doc,doc_dict[key]))
                    else:
                        postings[key] = [(i_doc, doc_dict[key])]
                    norm_factors[i_doc] += getTermFreq(doc_dict[key]) ** 2

                norm_factors[i_doc] = math.sqrt(norm_factors[i_doc])
                i_doc = i_doc + 1
    
    # Sorting the posting list for each term on the basis of document id
    for key in postings:
        postings[key] = sorted(postings[key])
    
    # Generate the index
    index = {
            'num_docs':num_docs,
            'postings':postings,
            'norm_factors':norm_factors,
            'doc_titles':doc_titles,
            'doc_ids':doc_ids
    }

    currentTime = datetime.now()
    print(f'Index construction took {timedelta_string(currentTime-startTime)}.')

    return index


def main():
    if len(sys.argv) != 2:
        print("Usage: python build_index.py <path to corpus root>")
        return

    print("Loading NLTK dependencies...")
    # Loading dependencies
    nltk.download('punkt')

    print(f"Building index for {sys.argv[1]}/AA/ ...")
    # Building the index
    index = buildIndex(sys.argv[1])

    """After generating the index with 89095 documents we then dump the index (pickle it) so that we can reuse it."""
    # Pickling the index
    pickle.dump(index, open('./index_new.pkl', 'wb'))
    print("Pickled index to ./index_new.pkl")

if __name__ == '__main__':
    main()