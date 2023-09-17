#!/usr/bin/env python
# coding: utf-8

# # Assignment -2 Experiment 2

# # Importing the Dependency 

# In[1]:


import os
import re
import math
import numpy as np
import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors
from tqdm import tqdm
from bs4 import BeautifulSoup  
from collections import Counter
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize


# # Preprocessing 
# 

# In[2]:


def preprocess(text):
    # Remove markup and non-text elements
    text = re.sub('<[^<]+?>', '', text)
    # Remove stopwords, filter out non-alphabetic words, and lowercase tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in TreebankWordTokenizer().tokenize(text) 
              if token.lower() not in stop_words and token.isalpha()]
    # Remove short words (length <= 2)
    tokens = [token for token in tokens if len(token) > 2]
    return tokens

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


documents


# # QUERY EXPANSION

# In[4]:


import gzip
import shutil
with gzip.open('GoogleNews-vectors-negative300.bin.gz', 'rb') as f_in:
    with open('GoogleNews-vectors-negative300.bin', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


# In[5]:


with open("test_50.txt", 'r') as f:
    contents = f.read()

soup = BeautifulSoup(str(contents), "html.parser")

query_dict = {}

for top in soup.find_all("top"):
    query_num = top.find("num").text
    query_title = top.find("title").text.strip()
    query_dict[query_num] = query_title

print(query_dict)


# In[6]:


model_path = 'GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Get a set of stopwords in English
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer('english')

# Define a regular expression pattern to remove punctuation and symbols
pattern = re.compile("[^\w\s]")

processed_queries = {}

for query_num, query_text in tqdm(query_dict.items()):
    
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
    
    # Remove words with length 2 or less
    query_stems = [stem for stem in query_stems if len(stem) > 2]

    # Expand the query using FastText embeddings
    expanded_query = []
    for token in query_tokens:
        # If the token is not in the embeddings model, add it as is to the expanded query
        if token not in w2v_model:
            expanded_query.append(token)
        else:
            # Get the top 3 most similar words to the token
            similar_words = w2v_model.most_similar(token, topn=3)
            for word, _ in similar_words:
                # If the similar word is not in the original query and is not a stopword or punctuation, add it to the expanded query
                if word not in query_tokens and word not in stop_words and not pattern.search(word):
                    expanded_query.append(word)
    
    # Add the expanded query to the processed_queries dictionary
    processed_queries[query_num] = list(set(query_stems + [stemmer.stem(token) for token in expanded_query]))


# In[7]:


processed_queries


# # Indexing

# In[8]:


# Initialize inverted index
inverted_index = defaultdict(list)

# Iterate over all documents
for doc_id, tokens in documents.items():
    # Create dictionary to store term frequencies in current 1document
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


# # Retrieval and Ranking

# In[10]:


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
with open('Results_expanedQuery.txt', 'w') as f:
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

        # Get the top 100 documents by cosine similarity for the current query
        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]

        # Write the top 10 results to the output file
        for i, (doc_id, score) in enumerate(top_docs):
            rank = i + 1
            line = f"{query_num} Q{query_num} {doc_id} {rank} {score:.4f} QueryExpansion\n"
            f.write(line)

            # Print the line to console (optional)
            print(line.strip())


# # MAP score and P@10 

# In[11]:


# initialize an empty dictionary of dictionary
relevance_judgment = {}

# open the document file for reading
with open('Relevance_judgments.txt', 'r') as f:
    # read each line of the file
    for line in f:
        # split the line by whitespace
        query_num, _, doc_name, bool_value = line.split()
        # extract the document name before the hyphen
        doc_name = doc_name.split('-')[0]
        # check if the query number is already in the dictionary
        if query_num not in relevance_judgment:
            relevance_judgment[query_num] = set()
        # add the document to the relevant set if bool_value is 1
        if bool_value == '1':
            relevance_judgment[query_num].add(doc_name)

# remove empty sets from the dictionary
relevance_judgment = {k: v for k, v in relevance_judgment.items() if v}

# print the dictionary of dictionary
print(relevance_judgment)


# In[12]:


# Compute average precision for a single query
def average_precision(documents, relevant_docs):
    relevant_count = 0
    precision_sum = 0
    for i, doc in enumerate(documents):
        if doc in relevant_docs:
            relevant_count += 1
            precision_sum += relevant_count / (i+1)
    if len(relevant_docs) == 0:
        return 0
    return precision_sum / len(relevant_docs)

# Compute Mean Average Precision (MAP) and precision at 10 (p@10) for all queries
MAP = 0
p_10 = 0
for query_num, query in processed_queries.items():
    # Compute cosine similarity scores between query and documents
    scores = defaultdict(int)
    for term in query:
        if term in inverted_index:
            for doc_id, tf in inverted_index[term]:
                scores[doc_id] += tf * math.log(N / df[term])
    scores = dict(scores)
    # Sort documents by decreasing similarity score
    documents_ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    # Compute average precision and precision at 10
    avg_precision = average_precision(documents_ranked, relevance_judgment[query_num])
    MAP += avg_precision
    p_10 += len(set(documents_ranked[:10]) & relevance_judgment[query_num]) / 10
MAP /= len(processed_queries)
p_10 /= len(processed_queries)


# In[13]:


#MAP score 
MAP


# In[14]:


#Precision at 10
p_10

