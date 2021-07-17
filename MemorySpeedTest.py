# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:55 2021

@author: MBo
"""
from RTArray import *
from time import time
from pympler import asizeof

test = RTArray('/home/mislav/Desktop/converting_plans/converted_plans/DIR_TEMP/xform_dir.mha')

print(asizeof.asizeof(test))

t1=time()
test.load_file()
t2=time()

print(asizeof.asizesof(test))
print()
print(t2-t1)