from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import kneighbors_graph
import numpy as np
from random import randrange, shuffle
import itertools
import sys
import random
import time
from time import sleep
import math
import operator
from matplotlib import pyplot as plt
import matplotlib as mpl
from time import sleep
from File_loader import load_data
from Evaluation import Evaluation_Metrics
from K_means import K_means
from Information_Class import Information

class Spectral_Clustering_GK:
    def __init__(self, dataset, number_of_data, data_distance_limit = 30, sigma = 1, k = 2):
        self.dataset = dataset
        self.number_of_data = number_of_data
        self.data_distance_limit = data_distance_limit
        self.sigma = sigma
        self.k = k
    
    def euclidean_distance(self, data1, data2, data_length):
        distance = 0
        for attribute in range(data_length):
            distance += pow((data1[attribute] - data2[attribute]), 2)
        return math.sqrt(distance)
    
    def gaussian_kernel(self, distance):
        return math.exp(-distance/2*self.sigma**2)

    def Affinity_Matrix(self):
        affinity_matrix = np.zeros((self.number_of_data, self.number_of_data))#pinakas opou kathe timh tou einai h apostash tou shmeioi i apo to j
        for i in range(self.number_of_data):
            for j in range(i):
                if i == j:#to dedomeno apo ton eauto tou exei profanws mhdenikh apostash
                    continue
                distance = self.euclidean_distance(self.dataset[i], self.dataset[j], len(self.dataset[i])-1)
                if distance < self.data_distance_limit:
                    value = self.gaussian_kernel(distance)
                    affinity_matrix[i][j] = value
                    affinity_matrix[j][i] = value
                    
                else:
                    affinity_matrix[i][j] = 0
                    affinity_matrix[j][i] = 0
        
        return affinity_matrix

    def Degree_Matrix(self, affinity_matrix):
        D = np.zeros((self.number_of_data, self.number_of_data))#pinakas opou kathe diagwnio stoixeio periexei to athroisma ths kathe grammhs tou affinity matrix
        sum_of_each_row = np.sum(affinity_matrix, axis=1)
        for i in range(self.number_of_data):
            D[i][i] = sum_of_each_row[i]
        return D
    
    def Laplacian_Matrix(self, A, D):
        return D-A

    def Normalized_Laplacian_Matrix(self, A, D):
        DM = np.copy(D)
        L = self.Laplacian_Matrix(A, D)
        for i in range(len(D)):
            DM[i][i] = 1.0/(D[i][i]*float(0.5))
        
        return DM.dot(L).dot(DM)

    def find_Eigenvalues_Eigenvectors(self, L):
        L = np.nan_to_num(L)
        #print(L)
        eigenvalues, eigenvectors = np.linalg.eig(L)
        #print(eigenvectors, eigenvalues.real)
        return eigenvalues.real, eigenvectors.real

    def find_maximum_eigenvalues(self, eigenvalues, k):
        maximum_indices = list()
        eigenvalues = list(eigenvalues)
        for i in range(k):#we want to find k eigenvectors for k clusters
            maximum = np.where(eigenvalues == np.amax(eigenvalues))#find the position of the max in list eigenvalues
            maximum_indices.append(maximum[0][0])#hold the position
            del eigenvalues[maximum[0][0]]#delete this max so we find the next
            
        return maximum_indices
            
    def choose_k(self, eigenvalues):
        
        eigengap = eigenvalues[2] - eigenvalues[1]       
        for i in range(3, eigenvalues.shape[0]):
            if eigenvalues[i] - eigenvalues[i-1] > eigengap:
                eigengap = eigenvalues[i] - eigenvalues[i-1]
                self.k = i
     
    def Spectral_Analysis_Transformation(self, L, eigenvalues, eigenvectors):
        #self.choose_k(eigenvalues)
        #print(self.k)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        #print(eigenvalues)
        eigenvectors = eigenvectors[:,idx]
        new_data = eigenvectors[:,-self.k:]
        #new_data = np.zeros((self.number_of_data, self.k))
        #indices = self.find_maximum_eigenvalues(eigenvalues, self.k)
        #for i in range(self.k):
        #    new_data[:,i] = eigenvectors[:,indices[i]]
        #print(new_data)
        
        return new_data
    
    def label_data(self, transformed_data):
        labels = list()
	for i in range(self.number_of_data):
            labels.append([self.dataset[i][len(self.dataset[i])-1]])
        return np.append(transformed_data, labels, axis=1)


    def show_graph(self):
        A = radius_neighbors_graph(self.dataset,0.4,mode='distance', metric='minkowski', p=2, metric_params=None, include_self=False)
        fig, ax = plt.subplots(figsize=(9,7))
        ax.set_title('dataset points', fontsize=18, fontweight='demi')
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1,1)
        ax.scatter(self.dataset[:, 0], self.dataset[:, 1],s= 5, cmap='viridis')
        for i in range(self.number_of_data):
            ax.annotate(i, (self.dataset[i,0],self.dataset[i,1]))
        plt.show()

    def plot_eigenvalues(self, eigenvalues):
        plt.title('Eigenvalues of Laplace Matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()
        plt.show()


    def implementation(self):
        print("Running Spectral Clustering algorithm...")
        sleep(2.0)
        start = time.time()
        A = self.Affinity_Matrix()
        D = self.Degree_Matrix(A)
        #L = self.Laplacian_Matrix(A, D)
        NL = self.Normalized_Laplacian_Matrix(A, D)
        eigenvalues, eigenvectors = self.find_Eigenvalues_Eigenvectors(NL)
        self.plot_eigenvalues(eigenvalues)
        transformed_data = self.Spectral_Analysis_Transformation(NL, eigenvalues, eigenvectors)
        #print(transformed_data)
	new_data = self.label_data(transformed_data)
	#print(new_data)
        classify = K_means(new_data, self.k , 1, "Spectral_Clustering")
        classify.implementation()
        end = time.time()
        
        
        
spambase = load_data("spambasetest.data")
spambase_data = spambase.load_data()
#spambase_data = spambase.shuffle_data()
s = Spectral_Clustering_GK(spambase_data, len(spambase_data))
#s.show_graph()
s.implementation()
