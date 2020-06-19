import numpy as np
from random import randrange, shuffle
import itertools
import random
import time
from time import sleep
import math
import operator
from matplotlib import pyplot as plt
from time import sleep
from File_loader import load_data
from Classify import classify
from Evaluation import Evaluation_Metrics
from Information_Class import Information
  
class K_means:
    def __init__(self, dataset, k=2, iterations=10, alg = "K_means"):
        self.k = k
        self.iterations = iterations
        self.dataset = dataset
        self.alg = alg
    
    
    def choose_centers(self):
        random_positions = random.sample(range(len(self.dataset)), self.k)
        centers = dict()
        for i in range(self.k):
            centers[i] = self.dataset[random_positions[i]]
	
        return centers

    def euclidean_distance(self, data1, data2, data_length):
        distance = 0
        for attribute in range(data_length):
            distance += pow((data1[attribute] - data2[attribute]), 2)
        return float(math.sqrt(distance))

    def algorithm_converged(self, previus_centers, new_centers):
        for center in new_centers:
            previus_center = previus_centers[center]
            new_center = new_centers[center]

            if self.euclidean_distance(previus_center, new_center, len(previus_center)-1) < 0.0001:
                return True
        return False

    '''
    def calculate2d_points(self, centers, clusters):
        centersmax = list()
        centersmin = list()
        clustersmin = list()
        clustersmax = list()
        for center in centers:
            maxc = np.max(centers[center])
            minc = np.min(centers[center])
            centersmax.append(maxc)
            centersmin.append(minc)
        for cluster in clusters:
            print(type(clusters[cluster]))
            for data in clusters[cluster]:
                maxc = np.max(data)
                minc = np.min(data)
                clustersmax.append(maxc)
                clustersmin.append(minc)
            
            
        return centersmax, centersmin, clustersmax, clustersmin  
            
    def plot_results(self, centersmax, centersmin, clustersmax, clustersmin):
        
        #for center in centers:
        #plt.figure()
        plt.scatter(centersmax, centersmin, marker='x')
        plt.scatter(clustersmax, clustersmin, marker='o')
        plt.show()
    '''       

    def compute_execution_time(self, start, end):
        final_execution_time = end - start
        final_execution_time = final_execution_time%3600/60
        return final_execution_time
        
        
    def implementation(self):
        results = list()#contain 10 dict of final clusterizations after 10 times run of kmeans algorithm
        for i in range(self.iterations):#run the kmeans 10 times with different centers each time
            centers = self.choose_centers()
            #print(centers)
            sleep(6.0)
            print("Running the K-Means algorithm %dth time" % (i+1))
            start = time.time()
            while True:
                clusters = dict()

                for cluster in range(self.k):
                    clusters[cluster] = list()

                for data in self.dataset:
                    distances = [self.euclidean_distance(data, centers[center],len(data) -1) for center in centers]
                    clusters[distances.index(min(distances))].append(data)

                previus_centers = dict(centers)
                
                for cluster in clusters:
                    centers[cluster] = np.average(clusters[cluster], axis=0)
                sleep(5.0)  
                if (self.algorithm_converged(previus_centers, centers)):
                    results.append(clusters)
                    break
            #print(clusters)
            end = time.time()
            execution_time = self.compute_execution_time(start, end)
            classification = classify(clusters, len(self.dataset))
            classes = classification.classification()
            metrics = Evaluation_Metrics(classes, len(self.dataset))
            purity = metrics.Purity()
            totalF_measure = metrics.TotalF_measure()
            information = Information(purity, totalF_measure, execution_time, self.alg)
            information.print_information()
           

spambase_data = load_data("spambasetest.data")
#spambase_data.show_data()
kmeans = K_means(spambase_data.load_data())
kmeans.implementation()





                
            
            
            
        
        
        
