from util import *

# Add your import statements here


class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		
		segmentedText = None
		
		#Fill in code here
		sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)

		segmentedText = sentences
		if sentences[-1].strip() == "": segmentedText = sentences[:-1]

		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		
		segmentedText = None
		
		#Fill in code here
		tokenizer = PunktSentenceTokenizer()
		tokenizer.train(text)
		segmentedLines = tokenizer.tokenize(text)

		segmentedText = []
		for s in segmentedLines:
			s = s[:-1].strip() if s[-1] in [".", "?", "!"] else s.strip()
			segmentedText.append(s)

		return segmentedText