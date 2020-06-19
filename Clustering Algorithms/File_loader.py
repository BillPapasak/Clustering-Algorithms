import numpy as np
from random import shuffle


class load_data:
    def __init__(self, file_name):
        self.file_name = file_name
		
    def load_data(self):
        print("Loading %s" % self.file_name)
        return np.loadtxt(self.file_name, delimiter=',')
    
    def show_data(self):
        data = self.load_data()
        print(data, len(data), type(data))
        
    def shuffle_data(self):
        dataset = list(self.load_data())
        shuffle(dataset)
        return np.array(dataset)
    
