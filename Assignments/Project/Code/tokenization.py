from util import *

# Add your import statements here
class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strinself, textgs where each string is a single sentence

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

		# Remove Punctuations and Split
		alphaRegex = re.compile("[^a-zA-Z]")
		tokenizedText = [[alphaRegex.sub(" ", token).lower().strip() for token in sentence] for sentence in tokenizedText]
		tokenizedTextOld = tokenizedText
		tokenizedText = []
		for sentence in tokenizedTextOld:
			sentenceRemoved = []
			for token in sentence:
				sentenceRemoved.extend(token.split())
			tokenizedText.append(sentenceRemoved)
		tokenizedText = [[token for token in sentence if not token == ""] for sentence in tokenizedText]

		return tokenizedText


	def ngram_tokenizer(self, text, ngram=2):
		"""
		Tokenization using ngram method
		
		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		arg2: integer
			The ngram size

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of ngram tokens
		"""
		
		tokenizedText = []
		
		#Fill in code here
		for sentence_tokens in text:
			tmp= zip(*[sentence_tokens[i:] for i in range(0, ngram)])
			result=[" ".join(ngram) for ngram in tmp]
			tokenizedText.append(result)

		return tokenizedText
