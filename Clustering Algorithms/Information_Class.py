from  time import sleep

class Information:
    def __init__(self, purity, F_measure, execution_time, alg = "None"):
        self.purity = purity
	self.F_measure = F_measure
        self.execution_time = execution_time
        self.alg = alg

    def print_information(self):
        print("Evaluating %s..."% self.alg)
        sleep(2.0)
        print("Purity = %lf" % self.purity)
        print("Total F_measure = %lf" % self.F_measure)
        print("Execution time  = %lf minutes" % self.execution_time)

