from pandas import qcut
from sqlalchemy import true
from util import *

# Add your import statements here




class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		relevant_docs = set(true_doc_IDs)
		predicted_docs = set(query_doc_IDs_ordered[:k])
		precision = Precision(relevant_docs, predicted_docs)

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		precisions = []
		# Get True Relevant Docs
		true_docs = GetQueryQRels(query_ids, qrels)
		true_docs = [[int(d["id"]) for d in q] for q in true_docs]

		# Calculate Mean Precision
		for i in range(len(query_ids)):
			relevant_docs = set(true_docs[i])
			predicted_docs = set(doc_IDs_ordered[i][:k])
			precisions.append(Precision(relevant_docs, predicted_docs))
		meanPrecision = (sum(precisions) + 0.0) / len(precisions)

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		relevant_docs = set(true_doc_IDs)
		predicted_docs = set(query_doc_IDs_ordered[:k])
		recall = Recall(relevant_docs, predicted_docs)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		recalls = []
		# Get True Relevant Docs
		true_docs = GetQueryQRels(query_ids, qrels)
		true_docs = [[int(d["id"]) for d in q] for q in true_docs]

		# Calculate Mean Recall
		for i in range(len(query_ids)):
			relevant_docs = set(true_docs[i])
			predicted_docs = set(doc_IDs_ordered[i][:k])
			recalls.append(Recall(relevant_docs, predicted_docs))
		meanRecall = (sum(recalls) + 0.0) / len(recalls)

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		relevant_docs = set(true_doc_IDs)
		predicted_docs = set(query_doc_IDs_ordered[:k])
		fscore = FScore(relevant_docs, predicted_docs)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		fscores = []
		# Get True Relevant Docs
		true_docs = GetQueryQRels(query_ids, qrels)
		true_docs = [[int(d["id"]) for d in q] for q in true_docs]

		# Calculate Mean FScore
		for i in range(len(query_ids)):
			relevant_docs = set(true_docs[i])
			predicted_docs = set(doc_IDs_ordered[i][:k])
			fscores.append(FScore(relevant_docs, predicted_docs))
		meanFscore = (sum(fscores) + 0.0) / len(fscores)

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		relevant_docs = true_doc_IDs
		predicted_docs = query_doc_IDs_ordered[:k]
		nDCG = NDCG(relevant_docs, predicted_docs, k)

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		ndcgs = []
		# Get True Relevant Docs
		true_docs = GetQueryQRels(query_ids, qrels)

		# Calculate Mean nDCG
		for i in range(len(query_ids)):
			relevant_docs = true_docs[i]
			predicted_docs = doc_IDs_ordered[i][:k]
			ndcgs.append(NDCG(relevant_docs, predicted_docs, k))
		meanNDCG = (sum(ndcgs) + 0.0) / len(ndcgs)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		relevant_docs = true_doc_IDs
		predicted_docs = query_doc_IDs_ordered[:k]
		avgPrecision = AveragePrecision(relevant_docs, predicted_docs)

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		averagePrecisions = []
		# Get True Relevant Docs
		true_docs = GetQueryQRels(query_ids, q_rels)
		true_docs = [[int(d["id"]) for d in q] for q in true_docs]

		# Calculate Mean Average Precision
		for i in range(len(query_ids)):
			relevant_docs = true_docs[i]
			predicted_docs = doc_IDs_ordered[i][:k]
			averagePrecisions.append(AveragePrecision(relevant_docs, predicted_docs))
		meanAveragePrecision = (sum(averagePrecisions) + 0.0) / len(averagePrecisions)

		return meanAveragePrecision

