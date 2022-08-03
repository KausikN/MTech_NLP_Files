# Add your import statements here
import re
import math
import numpy as np
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.corpus import wordnet, stopwords
nltk.download('stopwords')
nltk.download('universal_tagset')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Add any utility functions here
# Inflection Reduction
def GetWordNetPOS(tag):
    """
    Get Word Net POS tag from NLTK POS tag
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Information Retrieval
def Vectorise_Docs(docs):
    """
    Vectorise the documents
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    return tfidf_vectorizer,tfidf_matrix

def Vectorise_Query(vectorizer, merged_sentences):
    """
    Vectorise the query
    """
    tfidf_matrix = vectorizer.transform([merged_sentences])
    return tfidf_matrix

def GetSimilarity(query_vector, doc_vector):
    """
    Computes the cosine similarity between query and doc vectors
    """
    cosine_similarities = cosine_similarity(query_vector, doc_vector)
    return cosine_similarities

# Evaluation
def GetQueryQRels(query_ids, qrels):
    """
    Get the relevant documents for a given query
    """
    query_qrels = []
    for i in range(len(query_ids)):
        query_true_dicts = [qrel for qrel in qrels if int(qrel["query_num"]) == query_ids[i]]
        query_true_docs = []
        for qrel in query_true_dicts:
            query_true_docs.append(dict(qrel))
        query_qrels.append(query_true_docs)
    return query_qrels

def Precision(relevant_docs, retrieved_docs):
    """
    Computes Precision for a given query
    """
    if len(retrieved_docs) == 0:
        return 0
    return len(set(relevant_docs).intersection(set(retrieved_docs))) / len(retrieved_docs)

def Recall(relevant_docs, retrieved_docs):
    """
    Computes Recall for a given query
    """
    if len(relevant_docs) == 0:
        return 0
    return len(set(relevant_docs).intersection(set(retrieved_docs))) / len(relevant_docs)

def FScore(relevant_docs, retrieved_docs):
    """
    Computes FScore for a given query
    """
    precision = Precision(relevant_docs, retrieved_docs)
    recall = Recall(relevant_docs, retrieved_docs)
    if precision == 0 or recall == 0: return 0
    return (2 * precision * recall) / (precision + recall)

def GetRelevanceScore(relevant_docs, retrieved_docs):
    """
    Computes the relevance scores as 5 - position
    """
    scores = []
    for doc in retrieved_docs:
        score = 0
        for query in relevant_docs:
            if int(query["id"]) == doc:
                score = 5 - int(query["position"])
                break
        scores.append(score)
    return scores

def DCG(scores):
    """
    Computes DCG for given scores
    """
    dcg = 0.0
    for i in range(len(scores)):
        dcg += (((2**scores[i]) - 1) / (np.log2(i+2)))
    return dcg

def NDCG(relevant_docs_data, retrieved_docs, k):
    """
    Computes NDCG for a given query
    """
    retrieved_docs = list(retrieved_docs)
    if len(relevant_docs_data) == 0:
        return 0
    nDCG = 0.0
    
    # Get Query Relevances
    query_scores = GetRelevanceScore(relevant_docs_data, retrieved_docs)
    relevant_docs = []
    for relevant_doc in relevant_docs_data:
        relevant_docs.append(int(relevant_doc['id']))
    ideal_scores = GetRelevanceScore(relevant_docs_data, relevant_docs)
    # Calculate nDCG
    ideal_scores = sorted(ideal_scores, reverse=True)
    if not (ideal_scores[0] == 0):
        nDCG = DCG(query_scores) / DCG(ideal_scores[:k])

    return nDCG

def AveragePrecision(relevant_docs, retrieved_docs):
    """
    Computes Average Precision for a given query
    """
    retrieved_docs = list(retrieved_docs)
    if len(retrieved_docs) == 0 or len(relevant_docs)==0:
        return 0
    found = [
        int(retrieved_docs[i] in relevant_docs)
            for i in range(len(retrieved_docs))
        ]
    precisions = [
        Precision(relevant_docs, retrieved_docs[:i+1]) * found[i]
            for i in range(len(retrieved_docs))
        ]
    return sum(precisions) / (sum(found) + 1)