#!/usr/bin/env python
# coding: utf-8

# # Assignment 1 
# ## CSI4107 - Information Retrieval System
# 

# @author1: Rakshita Mathur
# 
# Student Number: 300215340
# 
# @author2: Fatimetou Fah
# 
# Student Number: 300101359

# #### Importing depandancy libraries

# In[1]:


import os
import re
import math
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup  
from collections import Counter
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


# # STEP1: Preprocessing 
# 
# Implement preprocessing functions for tokenization and stopword removal. The index terms will be all the words left after filtering out markup that is not part of the text, punctuation tokens, numbers, stopwords, etc. Optionally, you can use the Porter stemmer to stem the index words.
# 
# •       Input: Documents that are read one by one from the collection
# 
# •       Output: Tokens to be added to the index (vocabulary)

# In[2]:


def preprocess(text):
    # Remove markup and non-text elements
    text = re.sub('<[^<]+?>', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    # Remove punctuation, numbers, and short words (length <= 2)
    filtered_tokens = [token for token in filtered_tokens if token.isalpha() and len(token) > 2]
    return filtered_tokens

def stem_tokens(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

documents = {}
directory = "coll"
files = os.listdir(directory)
for file in tqdm(files):
    with open(os.path.join(directory, file), 'r') as f:
        text = f.read()
        tokens = preprocess(text)
        # stem the tokens
        stemmed_tokens = stem_tokens(tokens)
        # Save the filtered tokens to a new file
        documents[file] = stemmed_tokens


# In[3]:


#printing the tokenized dictonary of the collection of document.
documents 


# ##  Preprocessing Test Query

# In[4]:


with open("test_50.txt", 'r') as f:
    contents = f.read()

soup = BeautifulSoup(str(contents), "html.parser")

query_dict = {}

for top in soup.find_all("top"):
    query_num = top.find("num").text
    query_title = top.find("title").text.strip()
    query_dict[query_num] = query_title

print(query_dict)


# In[5]:


# Get a set of stopwords in English
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer('english')

# Define a regular expression pattern to remove punctuation and symbols
pattern = re.compile("[^\w\s]")

processed_queries = {}

for query_num, query_text in query_dict.items():
    
    #extract the number from the string
    
    query_num = re.findall(r'\d+', query_num)[0]
    
    # Tokenize the query text
    query_tokens = word_tokenize(query_text)
    
    # Remove stopwords and punctuation
    query_tokens = [token for token in query_tokens if token not in stop_words and not pattern.search(token)]
    
    # Remove newline characters
    query_tokens = [token.replace("\n", "") for token in query_tokens]
    
    # Stem the tokens
    query_stems = [stemmer.stem(token) for token in query_tokens]
    
    # Store the processed query
    processed_queries[query_num] = query_stems


# In[6]:


processed_queries


# # STEP 2 - INDEXING 
#  
# Build an inverted index, with an entry for each word in the vocabulary. You can use any appropriate data structure (hash table, linked lists, Access database, etc.). An example of possible index is presented below. Note: if you use an existing IR system, use its indexing mechanism.
# 
# •       Input: Tokens obtained from the preprocessing module
# 
# •       Output: An inverted index for fast access

# In[7]:


# Initialize inverted index
inverted_index = defaultdict(list)

# Iterate over all documents
for doc_id, tokens in documents.items():
    # Create dictionary to store term frequencies in current document
    term_freq = defaultdict(int)
    for token in tokens:
        # Increment term frequency for current document
        term_freq[token] += 1
    # Add (doc_id, term_freq) pairs to inverted index
    for token, freq in term_freq.items():
        inverted_index[token].append((doc_id, freq))

# Print inverted index
for term, postings in inverted_index.items():
    print(term + ': ' + str(postings))


# # STEP 3 Retrieval and Ranking 
# 
# Use the inverted index (from step 2) to find the limited set of documents that contain at least one of the query words. Compute the cosine similarity scores between a query and each document. 
# 
# • Input: One query and the Inverted Index (from Step2)
# 
# • Output: Similarity values between the query and each of the documents. Rank the documents in decreasing order of similarity scores.

# In[8]:


# Create a list of all unique terms across all documents
all_terms = set()
for tokens in documents.values():
    all_terms.update(tokens)

# Compute document frequencies (df) for all terms
df = Counter()
for tokens in documents.values():
    for term in set(tokens):
        df[term] += 1

# Compute idf for all terms
N = len(documents)
idf = {term: math.log(N/df[term]) for term in all_terms}

# Compute tf-idf vectors for all documents
tfidf_vectors = {}
for doc_id, tokens in documents.items():
    tf = Counter(tokens)
    tfidf = {term: (0.5 + 0.5 * tf[term]) * idf[term] for term in tf}
    tfidf_vectors[doc_id] = tfidf

# Loop over all queries
with open('Results.txt', 'w') as f:
    for query_num, query_tokens in processed_queries.items():
        # Compute tf-idf vector for the query
        query_tf = Counter(query_tokens)
        query_tfidf = {term: query_tf[term] * idf[term] for term in query_tf if term in idf}

        # Compute cosine similarity between the query and all documents
        scores = {}
        for doc_id, tfidf in tfidf_vectors.items():
            dot_product = sum(tfidf.get(term, 0) * query_tfidf.get(term, 0) for term in all_terms)
            doc_norm = math.sqrt(sum(val**2 for val in tfidf.values()))
            query_norm = math.sqrt(sum(val**2 for val in query_tfidf.values()))
            similarity = dot_product / (doc_norm * query_norm)
            scores[doc_id] = similarity

        # Get the top 10 documents by cosine similarity for the current query
        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

        # Write the top 10 results to the output file
        for i, (doc_id, score) in enumerate(top_docs):
            rank = i + 1
            line = f"{query_num} Q0 {doc_id} {rank} {score:.4f} \n"
            f.write(line)

            # Print the line to console (optional)
            print(line.strip())

