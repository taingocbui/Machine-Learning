import numpy as np
import sys
import load_data
import classifier as c
"""
This is the main python method that will be run.
You should determine what sort of command line arguments
you want to use. But in this module you will need to 
1) initialize your classifier and its params 
2) load training/test data 
3) train the algorithm
4) test it and output the desired statistics.
"""
result = []
(training, test) = load_data.load_congress_data()
c1 = c("naive_bayes")
c2 = c("neural")
c2.train(result)
c1.train(result)
c1.test(result)
