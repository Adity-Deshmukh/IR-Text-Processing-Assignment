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
from time import time

import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tokenize import regexp_tokenize

def getTermFreq(freq):
    '''
        int -> double

        Input : term_frequecny
        Output : normalized term frequency

        Description : 
        returns the log to the base 10 noramlized value 
    '''

    return 1 + math.log10(freq)

def getInvDocFreq(term, index):
    '''
        string,dict -> double

        Input : term,index
        Output : inverse document frequency

        Description : 
        returns the inverse document for the term
    '''

    num_docs = index['num_docs']
    doc_freq = len(index['postings'][term])
    return math.log10(num_docs / doc_freq)

def getTopKDocs(scores, k):
    '''
        list,int -> list

        Input : scores,k
        Output : list of documents

        Description : 
        returns a list of top 'k' documents with the highest score
    '''

    temp=list()

    for i in range(len(scores)):
        temp.append(i) 
    for n in range(0,k):
        maxi=n
        for i in range(n,len(temp)):
            if scores[temp[i]]>scores[temp[maxi]]:
                maxi=i
        temp1=temp[n]
        temp[n]=temp[maxi]
        temp[maxi]=temp1

    return temp[0:k]


def jaccard(title,query):
    '''
        string,string -> double

        Input : doc_title,query
        Output : double (jaccard coefficient)

        Description : 
        returns the jaccard coefficient of similarity between 
        document title and query term
    '''

    # Tokenizing the document title and query
    l1=regexp_tokenize(title.lower(), "[\w']+")
    l2=regexp_tokenize(query.lower(), "[\w']+")

    s1=set()
    s2=set()

    # Count for storing the intersection of title and query terms
    count=0
    for i in l1:
        s1.add(i)
        s2.add(i)
    
    for i in l2:
        if i in s1:
            s1.remove(i)
            count=count+1
        s2.add(i)
    
    return count/len(s2)

def expandTokenList(token_list, mask, index, expansion_factor=3, always_expand=False, discount_rate=0.1):
    '''
        list,list,dict,int,bool,double -> list,list

        Input : token_list, mask, index, expansion_factor, always_expand, discount_rate
        Output : exp_token_list,mask

        token_list : tokenised query terms
        mask : weight of original query terms (list of 1 * number of query terms)
        index : index
        expansion_factor : no of synonyms to be found out per term
        always_expand : specifies whether we want to use query expansion or not
        discount_rate : weight of the added terms
        
        exp_token_list : list of the query terms and their synonyms
        mask : modified mask with new values corresponding to the synonyms

        Description : 
        Computes the words most similar to the query words depending the arguments 
        provided, and returns the new query terms and modified mask

    '''

    exp_token_list = token_list.copy()

    for org_word in token_list:
        syn_count = 0

        # if the word is found in the index do not expand query corresponding to the 
        # term and check also check whether query expansion is true or not
        if org_word in index['postings'] and not always_expand:
            continue

        # Get lemma for the query term (equal to synset name)
        for synset in wordnet.synsets(org_word):
            lemma = synset.lemmas()[0] 

            # add synonym to the new list if the not already present in the list,
            # synonym should be present in the index and number of synonyms extracted
            # should be less than expansion factor
            if (syn_count < expansion_factor) and (lemma.name().lower() not in token_list) and (lemma.name().lower() in index['postings']):
                exp_token_list.append(lemma.name().lower())
                mask.append(discount_rate)
                syn_count += 1

    return exp_token_list, mask

def processQuery(query, index, k, titleWeighing=False, queryExpansion=False):
    '''
        string,dict,int,bool -> list

        Input : query,index,k,queryExpansion
        Output : scores

        Description : 
        returns the scores for the documents taking into account the query expansion
    '''
    
    # Tokenize the query after removing punctuations
    if titleWeighing:
        print(f"TITLE WEIGHING: ON")
    if queryExpansion:
        print("QUERY EXPANSION: ON")
    query = query.translate(str.maketrans('','',string.punctuation))
    scores = [0] * index['num_docs']
    query_freq = {}
    mask_dict = {}
    tokens = word_tokenize(query)
    tokens = [word.lower() for word in tokens]
    mask = [1] * len(tokens)

    # call query expansion
    if queryExpansion:
        tokens, mask = expandTokenList(tokens, mask, index, expansion_factor=5, always_expand=True, discount_rate=0.7)
        print(f"Expanded query: {str(tokens)}")

    # Generating query vector
    for i in range(len(tokens)):
        term = tokens[i]
        m_val = mask[i]

        if term in query_freq:
            query_freq[term] += 1
            mask_dict[term] = max(mask_dict[term], m_val)
        else:
            query_freq[term] = 1
            mask_dict[term] = m_val

    # Calculating cosine similarity and also introduced a mask factor which (generally)
    # reduces the weight of the synonyms taken into consideration
    query_norm = 0
    for term in query_freq:

        if term not in index['postings']:
            continue

        queryWeight = getTermFreq(query_freq[term]) * getInvDocFreq(term, index)
        query_norm += queryWeight ** 2
        for i_doc, docFreq in index['postings'][term]:
            docWeight = getTermFreq(docFreq)
            scores[i_doc] += queryWeight * docWeight * mask_dict[term]
    query_norm = math.sqrt(query_norm)
    if query_norm == 0:
        print("No query term in vocabulary. Retrieval failed.")
        return scores
    
    for i_doc, norm_factor in enumerate(index['norm_factors']):
        scores[i_doc] /= (norm_factor * query_norm)
        if titleWeighing:
            lambda1 = 0.5
            scores[i_doc]=scores[i_doc]+jaccard(index['doc_titles'][i_doc],query)*lambda1
    
    return scores

def testQuery(index, query, titleWeighing=False, queryExpansion=False):
    '''
        dict,string,bool -> 

        Input : index,query,queryExpansion
        Output : N/A

        Description : prints the top k documents with the highest (cosine) 
        similarity with the query
    '''
    # k to specify number of reults to get
    print(f'Retrieving documents for the query : {query}...')
    query = query.lower()
    k = 10
    start = time()
    scores = processQuery(query, index, k, titleWeighing=titleWeighing, queryExpansion=queryExpansion)
    res = getTopKDocs(scores, k)
    end = time()
    print(f"Query processing took {end-start:.3f} sec.")
    
    # Printing the results
    print('Results (Top-10):')
    for rank, i_doc in enumerate(res):
        doc_freq = -1
        print(f"{rank+1:3}: [DocID : {i_doc:5}] DocTitle : {index['doc_titles'][i_doc]:35} (Score : {scores[i_doc]:.3f})")



def main():
    if len(sys.argv) != 2:
        print("Usage: python test_queries.py <path to index.pkl file>")
        return

    # Load dependencies
    print("Loading NLTK dependencies...")
    nltk.download('punkt')
    nltk.download('wordnet')

    print(f"Unpickling {sys.argv[1]}...")
    # Loads the generated index
    index = pickle.load(open(sys.argv[1], 'rb'))    

    # Evaluating the query
    while True:
        query = input("Please enter the query to be searched : ")
        
        config = -1
        while not config in [0, 1, 2, 3]:
            config = int(input("Choose Query Processing configuration.\n0: Default (No improvements)\n1: Use Title Weighing\n2: Use Automatic Query Expansion\n3: Use both (1 and 2)\n> "))
            if config not in [0, 1, 2, 3]:
                print("Please choose a valid option (0/1/2/3).")

        titleWeighing = False
        queryExpansion = False
        
        if config == 1 or config == 3:
            titleWeighing = True
        if config == 2 or config == 3:
            queryExpansion = True
        testQuery(index, query, titleWeighing=titleWeighing, queryExpansion=queryExpansion)

        rep = input("Would you like to run another query? (Y/n)")
        if rep in ['n', 'N', "No"]:
            break

if __name__ == '__main__':
    main()