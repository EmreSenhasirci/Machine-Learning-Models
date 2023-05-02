# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 18:09:14 2023

@author: casper
"""
import numpy as np
import random 

null = 0
n=60000000
for i in range(n):
    sayi = random.randint(1,6)
    if sayi== 1:
        null +=sayi

oran = null/n

print(oran)    