from util import *

# Add your import statements here




class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.docs_map = {}

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = None

		#Fill in code here
		index = {}
		for doc_i in range(len(docs)):
			self.docs_map[docIDs[doc_i]] = docs[doc_i]
			doc = docs[doc_i]
			uniqueTermsInDoc = []
			for sentence in doc:
				uniqueTermsInDoc = uniqueTermsInDoc + sentence
			uniqueTermsInDoc = list(set(uniqueTermsInDoc))
			for term in uniqueTermsInDoc:
				if term not in index.keys():
					index[term.lower()] = [docIDs[doc_i]]
				else:
					index[term.lower()].append(docIDs[doc_i])

		self.index = index
		


	def rank(self, queries, **params):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
			[[['s1'],['s2']]]
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here
		# Doc Preprocess
		DOCS_PROCESSED = False
		docs = []
		doc_IDs = []
		doc_id_map = dict(self.docs_map)
		docs_embeddings = []

		# Query Rank
		for qi in tqdm(range(len(queries)), disable=False):
			query = queries[qi]
			terms = []
			merged_sentences = ""
			for sentence in query:
				merged_sentences = merged_sentences + " " + " ".join(sentence)
				terms.extend(sentence)
			terms = list(set(terms))

			# Get Vectorizer and Rank
			# Inverted Index Reduce Doc Set
			if params["inv_index_reduce"]:
				docs = []
				doc_IDs = []
				doc_id_map = {}
				for term in terms:
					term = term.lower()
					if term in self.index.keys():
						for d in self.index[term]:
							doc_id_map[d] = self.docs_map[d]
			if params["inv_index_reduce"] or not DOCS_PROCESSED:
				# Merge Documents
				for k in doc_id_map.keys():
					merged_doc = ""
					for sentence in doc_id_map[k]:
						merged_doc = merged_doc + " " + " ".join(sentence)
					docs.append(merged_doc)
					doc_IDs.append(k)

			# Construct Vectors and Rank
			if len(list(doc_id_map.keys())) >= 1:
				# Clean Docs and Query Words if using Word2Vec Model
				if params["model_clean_text"]:
					if params["inv_index_reduce"] or not DOCS_PROCESSED:
						docs = [Word2Vec_CleanText(params["Word2Vec_MODEL"], doc) for doc in docs]
					merged_sentences = Word2Vec_CleanText(params["Word2Vec_MODEL"], merged_sentences)

				# Get TFIDFs and Feature Name Maps
				if params["inv_index_reduce"] or not DOCS_PROCESSED:
					vectorizer, docs_tfidf_matrix = Vectorise_Docs_TFIDF(docs)
					docs_tfidf_matrix = docs_tfidf_matrix.toarray()
					feature_names = vectorizer.get_feature_names_out()
					feature_names_map = {feature_names[i]: i for i in range(len(feature_names))}
				query_tfidf_vector = Vectorise_Query_TFIDF(vectorizer, merged_sentences)
				query_tfidf_vector = query_tfidf_vector.toarray()

				# Apply Query Expansion Weights
				weights = params["sim_weights"][qi]
				weightsArray = []
				for f in feature_names: weightsArray.append(weights[f] if f in weights.keys() else 1.0)
				weightsArray = np.array(weightsArray)
				query_tfidf_eff_vector = np.multiply(query_tfidf_vector[0], weightsArray)
				query_tfidf_vector = query_tfidf_eff_vector
				
				# Get Query and Doc Vectors
				if params["vector_type"] == "BERT":
					# Get doc embeddings - embeddings for all docs(optimize by pickling)
					docs_final_matrix = params["BERT_doc_embeddings"]
					bert_model = params["BERT_MODEL"]
					query_final_vector = bert_model.encode(merged_sentences)
					# Calculate Similarity
					cosine_similarities = np.array(cosine_similarity([query_final_vector], docs_final_matrix))[0]

				elif params["vector_type"] == "TFIDF Stacking":
					query_final_vector = np.reshape(query_tfidf_vector, (1, -1))
					# Calculate Similarity
					cosine_similarities = GetSimilarity(query_final_vector, docs_tfidf_matrix)

				elif params["vector_type"] == "Word2Vec Without TFIDF":
					# Get Doc Word2Vec Vectors
					if params["inv_index_reduce"] or not DOCS_PROCESSED:
						docs_embeddings = []
						for doc in docs:
							doc_words = list(set(doc.split()))
							doc_words_vectors = np.array([
								Word2Vec_GetWordVector(params["Word2Vec_MODEL"], word) for word in doc_words
							])
							docs_embeddings.append(doc_words_vectors)
					# Get Query Word2Vec Vector
					query_words = list(set(merged_sentences.split()))
					query_embeddings = np.array([
						Word2Vec_GetWordVector(params["Word2Vec_MODEL"], word) for word in query_words
					])
					query_embeddings = np.array(query_embeddings)
					# Calculate Similarity
					cosine_similarities = []
					for doc_embeddings in docs_embeddings:
						sims = 0.0
						if 0 not in doc_embeddings.shape and 0 not in query_embeddings.shape:
							sims = np.array(cosine_similarity(query_embeddings, doc_embeddings))
							# [n_query_words, n_doc_words] -> single value
							sims = np.mean(np.mean(sims, axis=1), axis=0)
						cosine_similarities.append(sims)
					cosine_similarities = np.array(cosine_similarities)

				elif params["vector_type"] == "Word2Vec With TFIDF":
					# Get Doc Word2Vec Vectors
					if params["inv_index_reduce"] or not DOCS_PROCESSED:
						docs_embeddings = []
						for i in range(len(docs)):
							doc = docs[i]
							doc_tfidf_vector = docs_tfidf_matrix[i]
							doc_words = list(set(doc.split()))
							words_weightages = [
									doc_tfidf_vector[feature_names_map[word]] if word in feature_names_map.keys() 
									else doc_tfidf_vector.min() 
								for word in doc_words
							]
							doc_words_vectors = np.array([
								Word2Vec_GetWordVector(params["Word2Vec_MODEL"], doc_words[j]) * words_weightages[j] 
								for j in range(len(doc_words))
							])
							docs_embeddings.append(doc_words_vectors)
					# Get Query Word2Vec Vector
					query_words = list(set(merged_sentences.split()))
					words_weightages = [
							query_tfidf_vector[feature_names_map[word]] if word in feature_names_map.keys() 
							else query_tfidf_vector.min() 
						for word in query_words
					]
					query_embeddings = np.array([
						Word2Vec_GetWordVector(params["Word2Vec_MODEL"], query_words[j]) * words_weightages[j] 
						for j in range(len(query_words))
					])
					query_embeddings = np.array(query_embeddings)
					# Calculate Similarity
					cosine_similarities = []
					for doc_embeddings in docs_embeddings:
						sims = 0.0
						if 0 not in doc_embeddings.shape and 0 not in query_embeddings.shape:
							sims = np.array(cosine_similarity(query_embeddings, doc_embeddings))
							# [n_query_words, n_doc_words] -> single value
							sims = np.mean(np.mean(sims, axis=1), axis=0)
						cosine_similarities.append(sims)
					cosine_similarities = np.array(cosine_similarities)

				elif params["vector_type"] == "LSA":
					N_SINGULAR = 500
					# Apply LSA
					if params["inv_index_reduce"] or not DOCS_PROCESSED:
						U, S, VT = np.linalg.svd(docs_tfidf_matrix)
						eigvals = S**2 / np.sum(S**2)
						# Plot
						fig = plt.figure(figsize=(8,5))
						sing_vals = np.arange(docs_tfidf_matrix.shape[0]) + 1
						plt.plot(sing_vals, eigvals, "ro-", linewidth=2)
						plt.title("LSA Eigenvalues Plot")
						plt.xlabel("Principal Component")
						plt.ylabel("Eigenvalue")
						leg = plt.legend(["Eigenvalues from SVD"], loc="best")
						leg.get_frame().set_alpha(0.4)
						plt.savefig(os.path.join(params["output_dir"], "lsa_plot.png"))
						plt.close(fig)
						# Find Inverses and Project Query
						VT_inv = np.linalg.pinv(VT)
						S_inv = np.diag(np.reciprocal(S))
						docs_final_matrix = np.dot(U[:,:N_SINGULAR], np.diag(S)[:N_SINGULAR,:N_SINGULAR])
					
					query_final_vector = np.dot(query_tfidf_vector, np.dot(VT_inv[:,:N_SINGULAR], S_inv[:N_SINGULAR,:N_SINGULAR]))
					query_final_vector = np.reshape(query_final_vector, (1, -1))
					# Calculate Similarity
					cosine_similarities = GetSimilarity(query_final_vector, docs_final_matrix)

				elif params["vector_type"] == "Doc2Vec":
					doc2vec_model = params["Doc2Vec_MODEL"]
					# Find the vector of a Query
					query_final_vector = doc2vec_model.infer_vector(word_tokenize(merged_sentences))
					# Calculate Similarity
					cosine_similarities = doc2vec_model.docvecs.most_similar([query_final_vector])
					doc_ids = [int(sim_tuple[0]) + 1  for sim_tuple in cosine_similarities]
					doc_IDs_ordered.append(doc_ids)
					
				# Get Ranking
				if params["vector_type"] not in ["Doc2Vec"]:
					query_rank = [x for _, x in sorted(zip(cosine_similarities, doc_IDs), reverse=True)]
					doc_IDs_ordered.append(query_rank)
			else:
				doc_IDs_ordered.append([])
			
			# Update Progress
			params["progress_obj"]("Ranking: ", qi / len(queries))

			DOCS_PROCESSED = True
	
		return doc_IDs_ordered