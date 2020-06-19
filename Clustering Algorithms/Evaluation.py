class Evaluation_Metrics:
    def __init__(self, classes, N):
        self.classes = classes
	self.N = N
              
    def Purity(self):
        sumc = 0
        for c in self.classes:
            sumc += self.classes[c]['classData']
            
        return (1.0/float(self.N))*(sumc)

    def computeTP_FP_TN(self, cluster_class, class_id):
        #classes = self.data_class_count()
        
        TP = cluster_class['classData']
        FP = cluster_class['dataLength'] - cluster_class['classData']
        FN = 0
        '''
        for c in classes:
            if c != class_id and classes[c]['class'] != cluster_class['class']:
                FN += classes[c]['dataLength'] - classes[c]['classData']
        '''   
        #print(TP,FP)    
        return TP, FP, FN
        
    def computeRecall_precision(self, cluster_class, class_id):
        TP, FP, FN = self.computeTP_FP_TN(cluster_class, class_id)
        precision = float(TP)/float(TP+FP)
        recall = float(TP)/float(TP+FN)
	#print(precision, recall)
        return precision, recall

    def computeF_measure(self, cluster_class, class_id):
        precision, recall = self.computeRecall_precision(cluster_class, class_id)
        return 2.0/(1.0/precision+1.0/recall)
    
    def TotalF_measure(self):
        totalF_measure = 0
        for c in self.classes:
            totalF_measure += self.computeF_measure(self.classes[c], c)
        return totalF_measure
            
  
        
        
        
        
