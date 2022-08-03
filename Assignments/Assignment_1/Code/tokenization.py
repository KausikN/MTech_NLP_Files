from util import *

# Add your import statements here
import nltk
from nltk.tokenize import TreebankWordTokenizer



class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		#Fill in code here
		for sentence in text:
			sentence_tokens = sentence.split()
			tokenizedText.append(sentence_tokens)

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []
		
		#Fill in code here
		tokenizer = TreebankWordTokenizer()
		for sentence in text:
			sentence_tokens = tokenizer.tokenize(sentence)
			tokenizedText.append(sentence_tokens)

		return tokenizedText