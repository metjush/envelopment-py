"""
Sources:
Sherman & Zhu (2006) Sercice Productivity Management, Improving Service Performance using Data Envelopment Analysis (DEA) [Chapter 2]
ISBN: 978-0-387-33211-6
http://deazone.com/en/resources/tutorial

"""

import numpy as np
from scipy.optimize import fmin_slsqp


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

        # iterators
        self.unit_ = range(self.n)
        self.input_ = range(self.m)
        self.output_ = range(self.r)

        # result arrays
        self.output_w = np.zeros((self.r, 1), dtype=np.float)  # output weights
        self.input_w = np.zeros((self.m, 1), dtype=np.float)  # input weights
        self.lambdas = np.zeros((self.n, 1), dtype=np.float)  # unit efficiencies
        self.efficiency = np.zeros_like(self.lambdas)  # thetas

    def __efficiency(self):
        """
        Efficiency function to optimize
        :return: efficiency
        """

        # compute efficiency
        denominator = np.dot(self.inputs, self.input_w)
        numerator = np.dot(self.outputs, self.output_w)

        self.efficiency = numerator/denominator

    def __target(self, x, unit):
        """
        Theta target function for one unit
        :param x: combined weights
        :param unit:

        :return: theta
        """
        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m+self.r)], x[(self.m+self.r):]  # unroll the weights
        denominator = np.dot(self.inputs[unit], in_w)
        numerator = np.dot(self.outputs[unit], out_w)

        return numerator/denominator

    def __constraints(self, x, unit):
        """
        Constraints for optimization for one unit
        :param x: combined weights
        :param unit:
        :return: array of constraints
        """

        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m+self.r)], x[(self.m+self.r):]  # unroll the weights
        constr = []  # init the constraint array

        # for each input, lambdas with inputs
        for input in self.input_:
            t = self.__target(x, unit)
            lhs = np.dot(self.inputs[:, input], lambdas)
            cons = t*self.inputs[unit, input] - lhs
            constr.append(cons)

        # for each output, lambdas with outputs
        for output in self.output_:
            lhs = np.dot(self.outputs[:, output], lambdas)
            cons = lhs - self.outputs[unit, output]
            constr.append(cons)

        # for each unit
        for u in self.unit_:
            constr.append(lambdas[u])

        return np.array(constr)

    def __optimize(self):
        """
        Optimization of the DEA model
        Use: http://docs.scipy.org/doc/scipy-0.17.0/reference/generated/scipy.optimize.linprog.html
        A = coefficients in the constraints
        b = rhs of constraints
        c = coefficients of the target function
        :return:
        """
        d0 = self.m + self.r + self.n
        x0 = np.random.rand(d0) - 0.5
        # iterate over units
        for unit in self.unit_:
            # weights
            if unit != 4:
                continue
            x0 = fmin_slsqp(self.__target, x0, f_ieqcons=self.__constraints, args=(unit,))
        # unroll weights
        self.input_w, self.output_w, self.lambdas = x0[:self.m], x0[self.m:(self.m+self.r)], x0[(self.m+self.r):]

    def fit(self):
        """
        Optimize the dataset, generate basic table
        :return: table
        """

        self.__optimize()  # optimize
        self.__efficiency()  # compute efficiency

        print("Final thetas for each unit:\n")
        print("---------------------------\n")
        for n, eff in enumerate(self.lambdas):
            print("Unit %d theta: %.3f" % (n+1, eff))
            print("\n")
        print("---------------------------\n")


if __name__ == "__main__":
    X = np.array([
        [20., 300.],
        [30., 200.],
        [40., 100.],
        [20., 200.],
        [10., 400.]
    ])
    y = np.array([
        [1000.],
        [1000.],
        [1000.],
        [1000.],
        [1000.]
    ])
    dea = DEA(X,y)
    dea.fit()
