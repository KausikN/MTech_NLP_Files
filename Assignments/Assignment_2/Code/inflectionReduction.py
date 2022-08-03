from util import *

# Add your import statements here




class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = None

		#Fill in code here
		ReduceMethod = "lemmatization" # "stemming" or "lemmatization"

		if ReduceMethod == "stemming":
			# Stemming
			stemmer = nltk.stem.PorterStemmer()
			reducedText = [[stemmer.stem(token, to_lowercase=True) for token in sentence] for sentence in text]
		else:
			# Lemmatization
			pos_tags_sentences = [nltk.pos_tag(sentence) for sentence in text]
			pos_tags_sentences = [[GetWordNetPOS(tag) for (token, tag) in sentence] for sentence in pos_tags_sentences]
			lemmatizer = nltk.stem.WordNetLemmatizer()
			reducedText = [[lemmatizer.lemmatize(token, pos_tag) 
				for token, pos_tag in zip(sentence, pos_tags)] 
				for sentence, pos_tags in zip(text, pos_tags_sentences)]
		
		return reducedText