'''
Some necessary classes were implemented in this file

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


# Instance class, each sample is an instance
class Instance:
    __feature = []  # feature value in each dimension
    __fitness = 0  # fitness of objective function under those features

    def __init__(self, dim):
        self.__feature = []
        for i in range(dim.get_size()):
            self.__feature.append(0)
        self.__fitness = 0
        self.__dimension = dim

    def __str__(self):
        return 'Instance: feature ({}), fitness ({})'.format(self.__feature, self.__fitness)

    # return feature value in index-th dimension
    def get_feature(self, index):
        return self.__feature[index]

    # return features of all dimensions
    def get_features(self):
        return self.__feature

    # set feature value in index-th dimension
    def set_feature(self, index, v):
        self.__feature[index] = v

    # set features of all dimension
    def set_features(self, v):
        self.__feature = v

    # return fitness under those features
    def get_fitness(self):
        return self.__fitness

    # set fitness
    def set_fitness(self, fit):
        self.__fitness = fit

    #
    def equal(self, ins):
        if len(self.__feature) != len(ins.__feature):
            return False
        for i in range(len(self.__feature)):
            if self.__feature[i] != ins.__feature[i]:
                return False
        return True

    # copy this instance
    def copy_instance(self):
        copy = Instance(self.__dimension)
        for i in range(len(self.__feature)):
            copy.set_feature(i, self.__feature[i])
        copy.set_fitness(self.__fitness)
        return copy

    def show_instance(self):
        print('func-v:', self.__fitness, ' - ', self.__feature)


class FidelityInstance:
    __feature = []  # feature value in each dimension
    __lf_fitness = 0  # fitness of objective function under those features

    def __init__(self, dim):
        self.__feature = []
        for i in range(dim.get_size()):
            self.__feature.append(0)
        # for the instance which is not high evaluated, estimated_fitness = lf_eval + estimated_resdual
        # for the instance which is high evaluated, estimated_fitness = hf_eval
        # in optimization, rank of instances depends on estimated_fitness
        self.lf_fitness = 0.0
        self.hf_fitness = None
        self.estimated_fitness = 0.0  # estimated_eval = lf_eval + estimated_resdual
        self.__dimension = dim

    # return feature value in index-th dimension
    def get_feature(self, index):
        return self.__feature[index]

    # return features of all dimensions
    def get_features(self):
        return self.__feature

    # set feature value in index-th dimension
    def set_feature(self, index, v):
        self.__feature[index] = v

    # set features of all dimension
    def set_features(self, v):
        self.__feature = v

    def get_fitness(self):
        return self.estimated_fitness

    #
    def equal(self, ins):
        if len(self.__feature) != len(ins.__feature):
            return False
        for i in range(len(self.__feature)):
            if self.__feature[i] != ins.__feature[i]:
                return False
        return True

    # copy this instance
    def copy_instance(self):
        copy = FidelityInstance(self.__dimension)
        for i in range(len(self.__feature)):
            copy.set_feature(i, self.__feature[i])
        copy.lf_fitness = self.lf_fitness
        copy.hf_fitness = self.hf_fitness
        copy.estimated_fitness = self.estimated_fitness
        return copy

    def show_instance(self):
        print('func-v:', self.hf_fitness, ',', self.lf_fitness, ',', self.estimated_fitness, ' - ', self.__feature)


# Dimension class
# dimension message
class Dimension:
    __size = 0  # dimension size
    __region = []  # lower and upper bound in each dimension
    __type = []  # the type of each dimension, 0 is float, 1 is integer, 2 is categorical

    def __init__(self):
        return

    def set_dimension_size(self, s):
        self.__size = s
        self.__region = []
        self.__type = []

        for i in range(s):
            ori_reg = [0, 0]
            self.__region.append(ori_reg)
            self.__type.append(0)
        return

    # set index-th dimension region
    def set_region(self, index, reg, ty):
        self.__region[index][0] = reg[0]
        self.__region[index][1] = reg[1]
        self.__type[index] = ty
        return

    def set_regions(self, regs, tys):
        for i in range(self.__size):
            self.__region[i][0] = regs[i][0]
            self.__region[i][1] = regs[i][1]
            self.__type[i] = tys[i]
        return

    def get_size(self):
        return self.__size

    def get_region(self, index):
        return self.__region[index]

    def get_regions(self):
        return self.__region

    def get_type(self, index):
        return self.__type[index]
