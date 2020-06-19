class classify:

	def __init__(self, clusters, N):
		self.clusters = clusters
		self.N = N

	def classification(self):
		classes = dict()
		for cluster in self.clusters:
			spam_sum = 0
			not_spam_sum = 0
			classes[cluster] = dict()
			for data in self.clusters[cluster]:
				if data[len(data)-1] == 1.0:
					spam_sum += 1
				else:
					not_spam_sum += 1
			
			if spam_sum > not_spam_sum:
                
				classes[cluster]['class'] = 'spam'
				classes[cluster]['dataLength'] = len(self.clusters[cluster])
				classes[cluster]['classData'] = spam_sum
			else:
               
				classes[cluster]['class'] = 'not_spam'
				classes[cluster]['dataLength'] = len(self.clusters[cluster])
				classes[cluster]['classData'] = not_spam_sum
			#print(spam_sum, not_spam_sum, classes[cluster]['dataLength'], classes[cluster]['classData'])	

		return classes
