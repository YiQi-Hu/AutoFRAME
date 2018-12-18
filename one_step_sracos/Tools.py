'''
Some tools were implemented in this files

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

import random


# used for generating number randomly
class RandomOperator:

    def __init__(self):
        return

    def get_uniform_integer(self, lower, upper):
        return random.randint(lower, upper)

    def get_uniform_double(self, lower, upper):
        return random.uniform(lower, upper)


def list2string(list):
    my_str = str(list[0])
    i = 1
    while i < len(list):
        my_str = my_str + ' ' + str(list[i])
        i += 1
    return my_str
