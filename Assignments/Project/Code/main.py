# Imports
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from util import *

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt

# Main Vars
PRINT_OBJ = print
OUTPUT_DISPLAY_OBJ = None
PROGRESS_OBJ = None
PLOT_OBJ = None

TITLE_WEIGHTAGE = 1
QUERY_EXPAND_N = 0
# 0 - Sentence Segmentation
# 1 - Tokenization
# 2 - Inflection Reduction
# 3 - Stopword Removal
# 4 - NGram
# 5 - Query Expansion
QUERY_LOAD_POINT = 10
DOC_LOAD_POINT = 10

# Utils Functions
def Util_ProgressUpdate(text, progress):
    '''
    Progress Update
    '''
    if PROGRESS_OBJ is not None: PROGRESS_OBJ(text, progress)

def Util_OutputDisplayUpdate(key, data):
	'''
	Output Display Update
	'''
	if OUTPUT_DISPLAY_OBJ is not None: OUTPUT_DISPLAY_OBJ(key, data)

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args
		self.models = {}

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()

		self.informationRetriever = InformationRetrieval()
		self.evaluator = Evaluation()


	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)
		elif self.args.tokenizer == "ngram":
			return self.tokenizer.ngram_tokenizer(text, self.args.params["ngram_n"])

	def reduceInflection(self, text, ReduceMethod="lemmatization"):
		"""
		Call the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text, self.args.reducer)

	def removeStopwords(self, text):
		"""
		Call the required stopword remover
		"""
		return self.stopwordRemover.fromList(text, stopword_removal=self.args.params["stopword_removal"])

	def additionalPreprocessing(self, text):
		"""
		Call any additional preprocessing
		"""
		# NGram
		if self.args.params["ngram_n"] > 1:
			text = self.tokenizer.ngram_tokenizer(text, ngram=self.args.params["ngram_n"])
		return text

	def preprocessQueries(self, queries, **params):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""
		# LOAD
		LOAD_POINT = QUERY_LOAD_POINT
		# Segment queries
		if LOAD_POINT <= 0:
			segmentedQueries = []
			i = 0
			for query in queries:
				segmentedQuery = self.segmentSentences(query)
				# Spell Correction
				if params["spell_check"]:
					segmentedQuery = [SpellCorrect(sentence) for sentence in segmentedQuery]
				segmentedQueries.append(segmentedQuery)
				i += 1
				Util_ProgressUpdate("Query: Sentence Segmentation", i / len(queries))
			json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.json", 'w'), indent=4)
		elif LOAD_POINT == 1:
			segmentedQueries = json.load(open(self.args.out_folder + "segmented_queries.json", 'r'))
		# Tokenize queries
		if LOAD_POINT <= 1:
			tokenizedQueries = []
			i = 0
			for query in segmentedQueries:
				tokenizedQuery = self.tokenize(query)
				tokenizedQueries.append(tokenizedQuery)
				i += 1
				Util_ProgressUpdate("Query: Tokenization", i / len(segmentedQueries))
			json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.json", 'w'), indent=4)
		elif LOAD_POINT == 2:
			tokenizedQueries = json.load(open(self.args.out_folder + "tokenized_queries.json", 'r'))
		# Stem/Lemmatize queries
		if LOAD_POINT <= 2:
			reducedQueries = []
			i = 0
			for query in tokenizedQueries:
				reducedQuery = self.reduceInflection(query)
				reducedQueries.append(reducedQuery)
				i += 1
				Util_ProgressUpdate("Query: Inflection Reduction", i / len(tokenizedQueries))
			json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.json", 'w'), indent=4)
		elif LOAD_POINT == 3:
			reducedQueries = json.load(open(self.args.out_folder + "reduced_queries.json", 'r'))
		# Remove stopwords from queries
		if LOAD_POINT <= 3:
			stopwordRemovedQueries = []
			i = 0
			for query in reducedQueries:
				stopwordRemovedQuery = self.removeStopwords(query)
				stopwordRemovedQueries.append(stopwordRemovedQuery)
				i += 1
				Util_ProgressUpdate("Query: Stopword Removal", i / len(reducedQueries))
			json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.json", 'w'), indent=4)
		elif LOAD_POINT == 4:
			stopwordRemovedQueries = json.load(open(self.args.out_folder + "stopword_removed_queries.json", 'r'))
		# Form Final queries
		if LOAD_POINT <= 4:
			finalQueries = []
			i = 0
			for query in stopwordRemovedQueries:
				finalQuery = self.additionalPreprocessing(query)
				finalQueries.append(finalQuery)
				i += 1
				Util_ProgressUpdate("Query: Additional (NGram)", i / len(stopwordRemovedQueries))
			json.dump(finalQueries, open(self.args.out_folder + "final_queries.json", 'w'), indent=4)
		elif LOAD_POINT == 5:
			finalQueries = json.load(open(self.args.out_folder + "final_queries.json", 'r'))
		# Expand Queries
		if LOAD_POINT <= 5:
			expandedQueries = []
			expandedWeights = []
			i = 0
			for query in finalQueries:
				if QUERY_EXPAND_N > 0:
					expandedQuery, sim = QueryExpansion(params["Word2Vec_MODEL"], query, 0.1, n=QUERY_EXPAND_N)
				else:
					expandedQuery = query
					sim = {word: 1.0 for sentence in query for word in sentence}
				expandedQueries.append(expandedQuery)
				expandedWeights.append(sim)
				i += 1
				Util_ProgressUpdate("Query: Expansion", i / len(finalQueries))
			json.dump(expandedQueries, open(self.args.out_folder + "final_expanded_queries.json", 'w'), indent=4)
			json.dump(expandedWeights, open(self.args.out_folder + "final_expanded_queries_weights.json", 'w'), indent=4)
		else:
			expandedQueries = json.load(open(self.args.out_folder + "final_expanded_queries.json", 'r'))
			expandedWeights = json.load(open(self.args.out_folder + "final_expanded_queries_weights.json", 'r'))

		preprocessedQueries = expandedQueries
		out = {
			"weights": expandedWeights
		}
		return preprocessedQueries, out

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		# LOAD
		LOAD_POINT = DOC_LOAD_POINT
		# Segment docs
		if LOAD_POINT <= 0:
			segmentedDocs = []
			i = 0
			for doc in docs:
				segmentedDoc = self.segmentSentences(doc)
				segmentedDocs.append(segmentedDoc)
				i += 1
				Util_ProgressUpdate("Doc: Sentence Segmentation", i / len(docs))
			json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.json", 'w'), indent=4)
		elif LOAD_POINT == 1:
			segmentedDocs = json.load(open(self.args.out_folder + "segmented_docs.json", 'r'))
		# Tokenize docs
		if LOAD_POINT <= 1:
			tokenizedDocs = []
			i = 0
			for doc in segmentedDocs:
				tokenizedDoc = self.tokenize(doc)
				tokenizedDocs.append(tokenizedDoc)
				i += 1
				Util_ProgressUpdate("Doc: Tokenization", i / len(segmentedDocs))
			json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.json", 'w'), indent=4)
		elif LOAD_POINT == 2:
			tokenizedDocs = json.load(open(self.args.out_folder + "tokenized_docs.json", 'r'))
		# Stem/Lemmatize docs
		if LOAD_POINT <= 2:
			reducedDocs = []
			i = 0
			for doc in tokenizedDocs:
				reducedDoc = self.reduceInflection(doc)
				reducedDocs.append(reducedDoc)
				i += 1
				Util_ProgressUpdate("Doc: Inflection Reduction", i / len(tokenizedDocs))
			json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.json", 'w'), indent=4)
		elif LOAD_POINT == 3:
			reducedDocs = json.load(open(self.args.out_folder + "reduced_docs.json", 'r'))
		# Remove stopwords from docs
		if LOAD_POINT <= 3:
			stopwordRemovedDocs = []
			i = 0
			for doc in reducedDocs:
				stopwordRemovedDoc = self.removeStopwords(doc)
				stopwordRemovedDocs.append(stopwordRemovedDoc)
				i += 1
				Util_ProgressUpdate("Doc: Stopword Removal", i / len(reducedDocs))
			json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.json", 'w'), indent=4)
		elif LOAD_POINT == 4:
			stopwordRemovedDocs = json.load(open(self.args.out_folder + "stopword_removed_docs.json", 'r'))
		# Form Final docs
		if LOAD_POINT <= 4:
			finalDocs = []
			i = 0
			for doc in stopwordRemovedDocs:
				finalDoc = self.additionalPreprocessing(doc)
				finalDocs.append(finalDoc)
				i += 1
				Util_ProgressUpdate("Doc: Additional (NGram)", i / len(stopwordRemovedDocs))
			json.dump(finalDocs, open(self.args.out_folder + "final_docs.json", 'w'), indent=4)
		else:
			finalDocs = json.load(open(self.args.out_folder + "final_docs.json", 'r'))

		preprocessedDocs = finalDocs
		return preprocessedDocs

	def buildModels(self, docsData):
		"""
		Build Models
		"""
		# Load Docs
		processedDocs = docsData["docs_processed"]
		docs = docsData["docs"]
		docs_tokenized = docsData["docs_tokenized"]
		# Load Models
		model_params = {}
		if self.args.params["vector_type"] in ["Word2Vec Without TFIDF", "Word2Vec With TFIDF"] or \
				QUERY_EXPAND_N > 0:
			self.models["Word2Vec_MODEL"] = Word2Vec_BuildModel(processedDocs)
			model_params["Word2Vec_MODEL"] = self.models["Word2Vec_MODEL"]
		if self.args.params["vector_type"] in ["BERT"]:
			model_dir = os.path.join(self.args.out_folder, "models/")
			self.models["BERT_MODEL"], docEmbeddings = BERT_BuildModel(docs, model_dir)
			model_params["BERT_MODEL"] = self.models["BERT_MODEL"]
			model_params["BERT_doc_embeddings"] = docEmbeddings
		if self.args.params["vector_type"] in ["Doc2Vec"]:
			model_dir = os.path.join(self.args.out_folder, "models/")
			self.models["Doc2Vec_MODEL"] = Doc2Vec_BuildModel(processedDocs, model_dir)
			model_params["Doc2Vec_MODEL"] = self.models["Doc2Vec_MODEL"]
		if self.args.params["autocomplete"]:
			model_dir = os.path.join(self.args.out_folder, "models/")
			self.models["Autocomplete_MODEL"] = Autocomplete_BuildModel(docs_tokenized, model_dir)
			model_params["Autocomplete_MODEL"] = self.models["Autocomplete_MODEL"]

		return model_params

	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""
		global PLOT_OBJ
		# Read documents
		Util_ProgressUpdate("Dataset: Load and Clean", 0.0)
		docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
		# Read relevance judements
		qrels = json.load(open(self.args.dataset + "cran_qrels.json", 'r'))[:]
		# Clean Dataset
		docs_json, qrels = DatasetClean_RemoveEmptyDocs(docs_json, qrels)
		Util_ProgressUpdate("Dataset: Load and Clean", 1.0)

		# Split
		doc_ids = [item["id"] for item in docs_json]
		docs = [item["body"] for item in docs_json]
		docTitles = [item["title"] for item in docs_json]
		# Include Titles
		docs = [IncludeTitleInDoc(doc, title, TITLE_WEIGHTAGE) for doc, title in zip(docs, docTitles)]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Read queries
		queries_json = json.load(open(self.args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]

		# Build Models
		Util_ProgressUpdate("Model: Building Start", 0.0)
		docsData = {
			"docs": docs,
			"docs_processed": processedDocs,
			"docs_tokenized": json.load(open(self.args.out_folder + "tokenized_docs.json", 'r'))
		}
		model_params = self.buildModels(docsData)
		Util_ProgressUpdate("Model: Building End", 1.0)
		
		# Process queries
		processParams = self.args.params
		processParams.update(model_params)
		processParams.update({
			
		})
		processedQueries, queryData = self.preprocessQueries(queries, **processParams)

		# Build document index
		Util_ProgressUpdate("Ranking: Started", 0.0)
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for each query
		rankParams = self.args.params
		rankParams.update(model_params)
		rankParams.update({
			"output_dir": self.args.out_folder,
			"sim_weights": queryData["weights"],
			"progress_obj": Util_ProgressUpdate
		})
		doc_IDs_ordered = self.informationRetriever.rank(processedQueries, **rankParams)
		Util_ProgressUpdate("Ranking: Done", 1.0)

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
			fscores.append(fscore)
			PRINT_OBJ("Precision, Recall and F-score @ " +  
				str(k) + " : " + str(precision) + ", " + str(recall) + 
				", " + str(fscore))
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
			nDCGs.append(nDCG)
			PRINT_OBJ("MAP, nDCG @ " +  
				str(k) + " : " + str(MAP) + ", " + str(nDCG))

		# Tabulate the evaluations
		tableData = {
			"k": [i for i in range(1, len(precisions) + 1)],
			"Precision": precisions,
			"Recall": recalls,
			"F-score": fscores,
			"MAP": MAPs,
			"nDCG": nDCGs
		}
		Util_OutputDisplayUpdate("Evaluations", tableData)
		print("@1")
		print("nDCG: ", nDCGs[0], "MAP: ", MAPs[0])
		print("@10")
		print("nDCG: ", nDCGs[-1], "MAP: ", MAPs[-1])
		print("Max")
		print("nDCG: ", max(nDCGs), "MAP: ", max(MAPs))
		print()

		# Plot the metrics and save plot
		# Setup
		PLOT_OBJ = plt.figure()
		ax = plt.axes()
		ax.tick_params(axis='x', colors='blue')
		ax.tick_params(axis='y', colors='blue')
		# Plot
		plt.plot(range(1, 11), precisions, label="Precision")
		plt.plot(range(1, 11), recalls, label="Recall")
		plt.plot(range(1, 11), fscores, label="F-Score")
		plt.plot(range(1, 11), MAPs, label="MAP")
		plt.plot(range(1, 11), nDCGs, label="nDCG")
		# Legend
		legend = plt.legend()
		for text in legend.get_texts(): text.set_color("blue")
		# Title
		title = plt.title("Evaluation Metrics - Cranfield Dataset")
		title.set_color("blue")
		# Labels
		plt.xlabel("k")
		plt.ylabel("Metric")
		ax.xaxis.label.set_color("blue")
		ax.yaxis.label.set_color("blue")
		# Save
		plt.savefig(self.args.out_folder + "eval_plot.png")

		
	def handleCustomQuery(self, query=None):
		"""
		Take a custom query as input and return top five relevant documents
		"""
		# Read documents
		docs_json = json.load(open(self.args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids = [item["id"] for item in docs_json]
		docs = [item["body"] for item in docs_json]
		docTitles = [item["title"] for item in docs_json]
		# Include Titles
		docs = [IncludeTitleInDoc(doc, title, TITLE_WEIGHTAGE) for doc, title in zip(docs, docTitles)]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		#Get query
		if query is None:
			print("Enter query below")
			query = input()

		# Build Models
		Util_ProgressUpdate("Model: Building Start", 0.0)
		docsData = {
			"docs": docs,
			"docs_processed": processedDocs,
			"docs_tokenized": json.load(open(self.args.out_folder + "tokenized_docs.json", 'r'))
		}
		model_params = self.buildModels(docsData)
		Util_ProgressUpdate("Model: Building End", 1.0)

		# Process query
		processParams = self.args.params
		processParams.update(model_params)
		processParams.update({
			
		})
		processedQuery, queryData = self.preprocessQueries([query], **processParams)
		processedQuery = processedQuery[0]

		# Build document index
		Util_ProgressUpdate("Ranking: Started", 0.0)
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for the query
		rankParams = self.args.params
		rankParams.update(model_params)
		rankParams.update({
			"output_dir": self.args.out_folder,
			"sim_weights": queryData["weights"],
			"progress_obj": Util_ProgressUpdate
		})
		doc_IDs_ordered = self.informationRetriever.rank([processedQuery], **rankParams)[0]
		Util_ProgressUpdate("Ranking: Done", 1.0)

		# Print the IDs of first five documents
		PRINT_OBJ("\nTop five document IDs : ")
		for id_ in doc_IDs_ordered[:5]:
			PRINT_OBJ(id_)
		# Show the top five documents
		topDocs = []
		for i in doc_IDs_ordered[:5]:
			topDoc = {
				"id": i,
				"title": docs_json[doc_ids.index(i)]["title"],
				"body": docs_json[doc_ids.index(i)]["body"]
			}
			topDocs.append(topDoc)
		Util_OutputDisplayUpdate("Documents", topDocs)



if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-reducer',  default = "stemming",
	                    help = "Reducer Type [stemming|lemmatization]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()