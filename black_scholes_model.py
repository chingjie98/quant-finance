
import numpy as np
import math
import time

class OptionPricing:
    
	def __init__(self,S0,E,T,rf,sigma,iterations):
		self.S0 = S0
		self.E = E
		self.T = T
		self.rf = rf
		self.sigma = sigma     
		self.iterations = iterations 