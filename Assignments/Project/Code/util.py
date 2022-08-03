# Add your import statements here
import os
import re
import math
import string
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import TreebankWordTokenizer, word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.corpus import wordnet, stopwords
nltk.download('stopwords')
nltk.download('universal_tagset')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from tqdm import tqdm

# BERT
from sentence_transformers import SentenceTransformer, util
import pickle
from  gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
# Word2Vec
from gensim.models import Word2Vec
from gensim import corpora
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
# Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# Set Seed for Doc2Vec to remove randomization
hashseed = os.getenv("PYTHONHASHSEED")
if not hashseed:
    os.environ["PYTHONHASHSEED"] = "0"
# Autocomplete
from fast_autocomplete import AutoComplete

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
def Vectorise_Docs_TFIDF(docs):
    """
    Vectorise the documents
    """
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    return tfidf_vectorizer, tfidf_matrix

def Vectorise_Query_TFIDF(vectorizer, merged_sentences):
    """
    Vectorise the query
    """
    tfidf_matrix = vectorizer.transform([merged_sentences])
    return tfidf_matrix

def GetSimilarity(query_vector, doc_vector):
    """
    Computes the cosine similarity between query and doc vectors
    """
    # cosine_similarities = cosine_similarity(query_vector, doc_vector)
    cosine_similarities = []
    for doc in doc_vector:
        norm = np.linalg.norm(query_vector) * np.linalg.norm(doc)
        sim = 0.0
        if norm != 0.0:
            sim = np.dot(query_vector, doc.T) / norm
        cosine_similarities.append(sim)
    cosine_similarities = np.array(cosine_similarities)

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
        # dcg += (((2**scores[i]) - 1) / (np.log2(i+2)))
        dcg += ((scores[i]) / (np.log2(i+2)))
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
    # relevant_docs = []
    # for relevant_doc in relevant_docs_data:
    #     relevant_docs.append(int(relevant_doc['id']))
    # ideal_scores = GetRelevanceScore(relevant_docs_data, relevant_docs)
    ideal_scores = list(query_scores)
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
    # found = [
    #     int(retrieved_docs[i] in relevant_docs)
    #         for i in range(len(retrieved_docs))
    #     ]
    # precisions = [
    #     Precision(relevant_docs, retrieved_docs[:i+1]) * found[i]
    #         for i in range(len(retrieved_docs))
    #     ]
    # return sum(precisions) / len(relevant_docs)

    count = 0
    precision_total = 0.0
    for i in range(len(retrieved_docs)):
        if retrieved_docs[i] in relevant_docs:
            precision_total += ((count + 1) / (i + 1))
            count += 1
    if count > 0: precision_total = precision_total / count
    
    return precision_total

# Spelling Correction
def SpellCorrect(word):
    '''
    Spell Correction
    '''
    return str(TextBlob(str(word)).correct())

# Title Inclusion
def IncludeTitleInDoc(doc, title, weightage=1):
    '''
    Include the title in the document with weightage
    '''
    titleStr = " ".join([title] * weightage)
    return titleStr + ". " + doc

# Word 2 Vec
class Word2Vec_Corpus:
    def __init__(self, processedDocs):
        self.processedDocs = processedDocs

    def __iter__(self):
        for doc in self.processedDocs:
            yield doc

def Word2Vec_BuildModel(processedDocs):
    '''
    Build Word2Vec Model
    '''
    # corpus = Word2Vec_Corpus(processedDocs)
    corpus = []
    for doc in processedDocs: corpus.extend(doc)
    model = Word2Vec(corpus, min_count=5, vector_size=1024, workers=1, window=3, sg=0)
    return model

def Word2Vec_GetSimilarWords(model, word, n=5):
    '''
    Get Similar Words using Word 2 Vec
    '''
    if n <= 0: return []
    return model.wv.most_similar(word, topn=n)

def Word2Vec_GetWordVector(model, word):
    '''
    Get Word Vector using Word 2 Vec
    '''
    return model.wv[word]

def Word2Vec_CleanText(model, text):
    '''
    Clean Text by removing words not in model vocabulary
    '''
    return " ".join([word for word in text.split() if word in model.wv.index_to_key])

# Query Expansion
def QueryExpansion(model, query, simWeight=0.1, n=5):
    '''
    Expand Query using Word2Vec similarities
    '''
    # Combine query sentences
    query_merged = []
    for sentence in query: query_merged.extend(sentence)
    # Get Similar Words
    sim = {}
    query_exp = query_merged.copy()
    for q in query_merged:       
        sim[q] = 1
        try:
            sim_words = Word2Vec_GetSimilarWords(model, q, n)
            for w in sim_words:
                if w[0] not in query_exp:
                    query_exp.append(w[0])
                    # sim[w[0]] = w[1]
                    sim[w[0]] = simWeight
        except KeyError:
            pass
    query_exp = [query_exp]

    return query_exp, sim

# BERT
def BERT_BuildModel(docs, model_dir="output/models/"):
    '''
    Build BERT Model
    Multi QA Mini: multi-qa-MiniLM-L6-cos-v1
    Multi QA Base: multi-qa-mpnet-base-cos-v1
    '''
    MODEL_NAME = "multi-qa-mpnet-base-cos-v1"
    path_model = os.path.join(model_dir, MODEL_NAME + ".pkl")
    path_doc = os.path.join(model_dir, MODEL_NAME + "_" + "doc_embeddings_bert.pkl")
    # Check if already present
    if os.path.exists(path_doc) and os.path.exists(path_model):
        print("Loading BERT Model...")
        with open(path_doc, "rb") as f: doc_embeddings = pickle.load(f)
        with open(path_model, "rb") as f: model = pickle.load(f)
        return model, doc_embeddings
    # Else Load Model and Train
    model = SentenceTransformer(MODEL_NAME)
    doc_embeddings = model.encode(docs)
    with open(path_doc, "wb") as f: pickle.dump(doc_embeddings, f)
    with open(path_model, "wb") as f: pickle.dump(model, f)
    return model, doc_embeddings

# Doc2Vec
def Doc2Vec_BuildModel(docs, model_dir="output/models/"):
    '''
    Build Doc2Vec Model
    '''
    MODEL_NAME = "Doc2Vec"
    path_model = os.path.join(model_dir, MODEL_NAME + ".model")
    # path_model = os.path.join(model_dir, "enwiki_dbow/" + "doc2vec.bin")
    # Check if model is already present
    if os.path.exists(path_model):
        print("Loading Doc2Vec Model...")
        model = Doc2Vec.load(path_model)
        return model

    # Processed Docs
    doc_list = []
    for doc in docs:
        merged_sentences = ""
        for sentence in doc:
            merged_sentences = merged_sentences + " " + " ".join(sentence)
        doc_list.append(merged_sentences.strip())
    # Load Model and Train
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(doc_list)]
    vec_size = 1024
    model = Doc2Vec(vector_size=vec_size, workers=1, seed=1, min_count=1, dm=0)
    # Build Model Vocabulary
    model.build_vocab(tagged_data)
    # Train model on Docs
    model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=50)
    # Save Model
    model.save(path_model)
    
    return model

# Autocomplete
def Autocomplete_BuildModel(docs, model_dir="output/models/"):
    '''
    Build Autocomplete Model
    '''
    # Docs
    words = [w for doc in docs for sentence in doc for w in sentence]
    wordsDict = {}
    for w in words:
        wordsDict[w] = {}
    # Load Model and Train
    model = AutoComplete(words=wordsDict)
    
    return model

# Dataset Cleaning
def DatasetClean_RemoveEmptyDocs(docs, qrels):
    '''
    Remove Empty Documents
    '''
    # Remove doc if empty and remove in qrels also
    docs_clean = []
    qrels_clean = qrels
    print("Before Cleaning: Docs: {}, Qrels: {}".format(len(docs), len(qrels)))
    for i in range(len(docs)):
        doc = docs[i]
        if doc["title"].strip() == "" and doc["body"].strip() == "":
            for qrel in qrels_clean:
                if str(qrel["id"]) == str(doc["id"]):
                    qrels_clean.remove(qrel)
        else:
            docs_clean.append(doc)
    print("After Cleaning: Docs: {}, Qrels: {}".format(len(docs_clean), len(qrels_clean)))
            
    return docs_clean, qrels_clean

# Main Vars