"""
Sources:
Sherman & Zhu (2006) Sercice Productivity Management, Improving Service Performance using Data Envelopment Analysis (DEA) [Chapter 2]
ISBN: 978-0-387-33211-6
http://deazone.com/en/resources/tutorial

"""

import numpy as np


class DEA(object):

    def __init__(self, inputs, outputs):
        """
        Initialize the DEA object with input data
        n = number of entities (observations)
        m = number of inputs (variables, features)
        r = number of outputs
        :param inputs: inputs, n x m numpy array
        :param outputs: outputs, n x r numpy array
        :return: self
        """

        # supplied data
        self.inputs = inputs
        self.outputs = outputs

        # parameters
        self.n = inputs.shape[0]
        self.m = inputs.shape[1]
        self.r = outputs.shape[1]

        # result arrays
        self.output_w = np.zeros((self.r, 1), dtype=np.float) # output weights
        self.input_w = np.zeros((self.m, 1), dtype=np.float) # input weights
        self.efficiency = np.zeros((self.n, 1), dtype=np.float) # unit efficiencies

    def __efficiency(self, out_w, in_w):
        """
        Efficiency function to optimize
        :param out_w: output weights
        :param in_w: input weights
        :return: efficiency
        """

        denominator = np.dot(self.inputs, in_w)
        numerator = np.dot(self.outputs, out_w)

        return numerator/denominator



