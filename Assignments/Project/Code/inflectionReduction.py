from util import *

# Add your import statements here




class InflectionReduction:

	def reduce(self, text, ReduceMethod="lemmatization"):
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
		curText = text

		# Lemmatization
		if ReduceMethod == "lemmatization" or ReduceMethod == "both":
			# print('inside lemm')
			pos_tags_sentences = [nltk.pos_tag(sentence) for sentence in curText]
			pos_tags_sentences = [[GetWordNetPOS(tag) for (token, tag) in sentence] for sentence in pos_tags_sentences]
			lemmatizer = nltk.stem.WordNetLemmatizer()
			curText = [[lemmatizer.lemmatize(token, pos_tag) 
				for token, pos_tag in zip(sentence, pos_tags)] 
				for sentence, pos_tags in zip(curText, pos_tags_sentences)]

		# Stemming
		if ReduceMethod == "stemming" or ReduceMethod == "both":
			# stemmer = nltk.stem.PorterStemmer()
			stemmer = nltk.stem.SnowballStemmer("english",ignore_stopwords=True)
			for i in range(2):
				curText = [[stemmer.stem(token.lower()) for token in sentence] for sentence in curText]


		reducedText = curText
		
		return reducedText