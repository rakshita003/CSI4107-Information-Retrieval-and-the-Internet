#!/usr/bin/env python
# coding: utf-8

# # Assignment -2 Experiment 1

# In[1]:


import os
import re
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import word_tokenize
from collections import defaultdict
import math
from tqdm import tqdm
from bs4 import BeautifulSoup
import nltk
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


# ## Doc2vec

# In[2]:


# Function to preprocess text for doc2vec
def preprocess_doc2vec(text):
    # Remove markup and non-text elements
    text = re.sub('<[^<]+?>', '', text)
    # Tokenize text at the document level
    tokens = [word_tokenize(sent) for sent in nltk.sent_tokenize(text)]
    # Remove stopwords and short tokens (length <= 2)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [[token.lower() for token in sent if token.lower() not in stop_words and len(token) > 2] for sent in tokens]
    return filtered_tokens

documents = {}
directory = "coll"
files = os.listdir(directory)
for file in tqdm(files):
    with open(os.path.join(directory, file), 'r') as f:
        text = f.read()
        tokens = preprocess_doc2vec(text)
        # Save the filtered tokens to a new file
        documents[file] = tokens

with open("test_50.txt", 'r') as f:
    contents = f.read()

soup = BeautifulSoup(str(contents), "html.parser")

query_dict = {}

for top in soup.find_all("top"):
    query_num = top.find("num").text
    query_title = top.find("title").text.strip()
    query_dict[query_num] = query_title

print(query_dict)

# Get a set of stopwords in English
stop_words = set(stopwords.words("english"))

processed_queries = {}

for query_num, query_text in query_dict.items():
    
    #extract the number from the string
    
    query_num = re.findall(r'\d+', query_num)[0]
    
    # Tokenize the query text
    query_tokens = word_tokenize(query_text)
    
    # Remove stopwords and punctuation
    query_tokens = [token for token in query_tokens if token not in stop_words]
    
    # Stem the tokens
    query_tokens = [token.lower() for token in query_tokens if len(token) > 2]
    
    # Store the processed query
    processed_queries[query_num] = query_tokens 



# In[3]:


# Create tagged documents for doc2vec
tagged_documents = [TaggedDocument([word for sublist in doc for word in sublist], [doc_id]) for doc_id, doc in documents.items()]


# Train doc2vec model

model = Doc2Vec(vector_size=300, window=5, min_count=5, alpha=0.025, min_alpha=0.001, epochs=40)
model.build_vocab(tagged_documents)
model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)


# Loop over all queries
with open('Results_Doc2vec.txt', 'w') as tqdm(f):
    for query_num, query_tokens in processed_queries.items():
        # Get the inferred vector for the query
        query_vector = model.infer_vector(query_tokens)

        # Compute cosine similarity between the query and all documents
        scores = {}
        for doc_id in documents.keys():
            doc_vector = model.docvecs[doc_id]
            similarity = cosine_similarity([query_vector], [doc_vector])[0][0]
            scores[doc_id] = similarity

        # Get the top 1000 most similar documents
        top_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]

        # Write the results to the output file
        for rank, (doc_id, score) in enumerate(top_matches):
            f.write(f'{query_num} Q0 {doc_id} {rank+1} {score:.6f} Doc2Vec\n')



# ## MAP score and P@10

# In[4]:


# initialize an empty dictionary of dictionary
relevance_judgment = {}

# open the document file for reading
with open('Relevance_judgments.txt', 'r') as f:
    # read each line of the file
    for line in tqdm(f):
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


# In[5]:


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

MAP = 0
p_10 = 0
for query_num, query in tqdm(processed_queries.items()):
    # Get document vectors for all documents in the corpus
    document_vectors = [model.docvecs[doc_id] for doc_id in model.docvecs.index2entity]

    # Get the query vector
    query_vector = model.infer_vector(query)
    # Compute cosine similarity scores between query and documents
    scores = {}
    for i, doc_vector in enumerate(document_vectors):
        score = cosine_similarity([query_vector], [doc_vector])[0][0]
        doc_id = model.docvecs.index2entity[i]
        scores[doc_id] = score

    # Sort documents by decreasing similarity score
    documents_ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    # Compute average precision and precision at 10
    avg_precision = average_precision(documents_ranked, relevance_judgment[query_num])
    MAP += avg_precision
    p_10 += len(set(documents_ranked[:10]) & relevance_judgment[query_num]) / 10
MAP /= len(processed_queries)
p_10 /= len(processed_queries)


# In[6]:


MAP


# In[7]:


p_10

