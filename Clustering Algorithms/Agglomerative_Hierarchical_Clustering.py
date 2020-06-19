import numpy as np
from random import randrange, shuffle
import itertools
import random
from time import sleep
import time
import math
import operator
from matplotlib import pyplot as plt
from time import sleep
from File_loader import load_data
from Classify import classify
from Evaluation import Evaluation_Metrics
from Information_Class import Information

class Agglomerative_Hierarchical_Clustering:
    def __init__(self, dataset, number_of_data, number_of_clusters=2):
        self.dataset = dataset
        self.number_of_data = number_of_data
        self.clusters = dict()
	self.number_of_clusters = number_of_clusters
	
    
    def euclidean_distance(self, data1, data2, data_length):
        distance = 0
        for attribute in range(data_length):
            distance += pow((data1[attribute] - data2[attribute]), 2)
        return math.sqrt(distance)

    def initialization(self):
        for i in range(self.number_of_data):
            self.clusters[i] = list()
            self.clusters[i].append(self.dataset[i])
        

    def find_middle_representative(self, cluster_data):
        if len(cluster_data) == 1:#an to cluster periexei mono ena dianusma tote profanws to meso einai to idio to dianusma
            return cluster_data[0]
        else:
            return np.average(cluster_data, axis=0)

    def distance_of_clusters(self, fcluster_data, scluster_data):
        r1 = self.find_middle_representative(fcluster_data)
        r2 = self.find_middle_representative(scluster_data)

        return self.euclidean_distance(r1, r2, len(r1)-1)
    
    def delete_cluster(self, cluster_id):
        del self.clusters[cluster_id]

    def merge_clusters(self, fcluster_id, scluster_id):
        self.clusters[fcluster_id].extend(self.clusters[scluster_id])
        self.delete_cluster(scluster_id)
    
    def compute_execution_time(self, start, end):
        final_execution_time = end - start
        final_execution_time = final_execution_time%3600/60
        return final_execution_time
    
    
    def implementation(self):
        number_of_clusters = len(self.clusters)
	level = 0
        self.initialization()
        print("Running Agglomerative Hierarchical Clustering algorithm...")
        start = time.time()
        sleep(2.0)
        while len(self.clusters) != self.number_of_clusters:
            c = []
            print("Level %d..."% level)
            for i, j in itertools.combinations(self.clusters, 2):
                distance = self.distance_of_clusters(self.clusters[i], self.clusters[j])
                c.append([distance, i, j])
            min_distance = min(c, key=lambda c: c[0])
            #print(min_distance)
            self.merge_clusters(min_distance[1], min_distance[2])
            level += 1
        end = time.time()
        execution_time = self.compute_execution_time(start, end)
        classification = classify(self.clusters, self.number_of_data)
        classes = classification.classification()
        metrics = Evaluation_Metrics(classes, self.number_of_data)
        purity = metrics.Purity()
        totalF_measure = metrics.TotalF_measure()
        information = Information(purity, totalF_measure, execution_time, "Agglomerative_Hierarchical_Clustering")
        information.print_information()
    
spambase = load_data("spambasetest.data")
spambase_data = spambase.load_data()
#spambase.show_data()
#spambase_data = spambase.shuffle_data()
#spambase.show_data()
k = Agglomerative_Hierarchical_Clustering(spambase_data, len(spambase_data))
k.implementation()
    


                












            










        
        
        
            
