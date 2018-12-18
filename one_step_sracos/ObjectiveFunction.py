'''
Objective functions can be implemented in this file

Author:
    Yi-Qi Hu

Time:
    2016.6.13
'''

'''
 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

 Copyright (C) 2015 Nanjing University, Nanjing, China
'''

import math
from one_step_sracos.Tools import RandomOperator


# Sphere function for continue optimization
def Sphere(x):
    value = sum([(i - 0.2) * (i - 0.2) for i in x])
    return value


# Ackley function for continue optimization
def Ackley(x):
    bias = 0.04
    value_seq = 0
    value_cos = 0
    for i in range(len(x)):
        value_seq += (x[i] - bias) * (x[i] - bias)
        value_cos += math.cos(2.0 * math.pi * (x[i] - bias))
    ave_seq = value_seq / len(x)
    ave_cos = value_cos / len(x)
    value = -20 * math.exp(-0.2 * math.sqrt(ave_seq)) - math.exp(ave_cos) + 20.0 + math.e
    return value


# A test function for mixed optimization
def MixedFunction(x):
    value = sum([i * i for i in x])
    return value


# set cover problem for discrete optimization
def SetCover(x):
    weight = [0.8356, 0.5495, 0.4444, 0.7269, 0.9960, 0.6633, 0.5062, 0.8429, 0.1293, 0.7355,
              0.7979, 0.2814, 0.7962, 0.1754, 0.0267, 0.9862, 0.1786, 0.5884, 0.6289, 0.3008]
    subset = []
    subset.append([0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0])
    subset.append([0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0])
    subset.append([1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0])
    subset.append([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    subset.append([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1])
    subset.append([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    subset.append([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0])
    subset.append([0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0])
    subset.append([0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0])
    subset.append([0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1])
    subset.append([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0])
    subset.append([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1])
    subset.append([1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1])
    subset.append([1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1])
    subset.append([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
    subset.append([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0])
    subset.append([1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1])
    subset.append([0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1])
    subset.append([0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0])
    subset.append([0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1])

    allweight = 0
    countw = 0
    for i in range(len(weight)):
        allweight = allweight + weight[i]

    dims = []
    for i in range(len(subset[0])):
        dims.append(False)

    for i in range(len(subset)):
        if x[i] == 1:
            countw = countw + weight[i]
            for j in range(len(subset[i])):
                if subset[i][j] == 1:
                    dims[j] = True
    full = True
    for i in range(len(dims)):
        if dims[i] == False:
            full = False

    if full is False:
        countw = countw + allweight

    return countw


class DistributedFunction:

    def __init__(self, dim):

        ro = RandomOperator()

        self.__dimension = dim
        # generate bias randomly
        self.__bias = []
        for i in range(self.__dimension.getSize()):
            # self.__bias.append(ro.getUniformDouble(self.__dimension.getRegion(i)[0], self.__dimension.getRegion(i)[1]))
            self.__bias.append(ro.getUniformDouble(-0.5, 0.5))
        # print 'bias:', self.__bias

        return

    def getBias(self):
        return self.__bias

    def setBias(self, b):
        self.__bias = []
        for i in range(len(b)):
            self.__bias.append(b[i])
        # print 'bias:', self.__bias
        return

    def DisAckley(self, x):
        value_seq = 0
        value_cos = 0
        for i in range(len(x)):
            value_seq += (x[i] - self.__bias[i]) * (x[i] - self.__bias[i])
            value_cos += math.cos(2.0 * math.pi * (x[i] - self.__bias[i]))
        ave_seq = value_seq / len(x)
        ave_cos = value_cos / len(x)
        value = -20 * math.exp(-0.2 * math.sqrt(ave_seq)) - math.exp(ave_cos) + 20.0 + math.e
        return value

    def DisSphere(self, x):
        value = 0
        for i in range(len(x)):
            value += (x[i] - self.__bias[i]) * (x[i] - self.__bias[i])
        return value
