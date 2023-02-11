# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:42:28 2022

@author: Elia Friberg
"""

#lab 1
#1.1
import sys
help(sys.float_info)
print(sys.float_info)
#max è il maggior numero rappresentabile
#max_esp l'esponente maggiore in base 2 rappresentabile
#max_10_esp uguale ma in base 10
#1.2

#epsilon di macchina
epsilon = 1.0
mant_dig = 1
while 1.0 + epsilon / 2.0 > 1.0:
    epsilon = epsilon/2.0
    mant_dig = mant_dig+1
print ("epsilon = " + str(epsilon))
print ("mant_dig = " + str(mant_dig), '\n')

#1.3
#epsilon con diversi float
import numpy as np
print ("float16: ")
epsilon = np.float16(1.0)
mant_dig = 1.0
while np.float16(1.0) + epsilon / np.float16(2.0) > np.float16(1.0):
    epsilon = epsilon/np.float16(2.0)
    mant_dig = mant_dig+1
print ("  epsilon = " + str(epsilon))
print ("  mant_dig = " + str(mant_dig),'\n')

print ("float32: ")
epsilon = np.float32(1.0)
mant_dig = 1
while np.float32(1.0) + epsilon / np.float32(2.0) > np.float32(1.0):
    epsilon =epsilon/ np.float32(2.0)
    mant_dig = mant_dig+1
print ("  epsilon = " + str(epsilon))
print ("  mant_dig = " + str(mant_dig),'\n')                                                                                                                         

print(np.finfo(float).eps)

#2.1

import numpy as np
import matplotlib.pyplot as plt
#sin e cos

def n1():
    linspace = np.linspace(0, 10)
    plt.plot(linspace, np.sin(linspace),label='sin', color='yellow')
    plt.plot(linspace, np.cos(linspace),label='cos', color='blue')
    plt.legend(loc='upper right')
    plt.grid()
    plt.title('seno e coseno da 0 a 10')
    plt.show()


###2.2
def n2(n):
    #fibonacci<n
    if n <= 0:
        return 0
    if n <= 1:
        return 1
    a, b = 0, 1
    cont = 2
    while a+b < n:
        b = b + a
        a = b - a
        cont = cont + 1
    return cont

def n3():
    #errore relativo
    arange = np.arange(50)
    plt.plot(arange, [relative_error(i) for i in arange], color='red',label='Errore Relativo') #LAMBDA
    plt.legend(loc='upper right')
    plt.grid()
    plt.show() 
    #già quando x = 10 la funzione tende a 0

def r(k): # assuming k > 0
#ratio di fibonacci
    if k <= 1: return 0                                                                                                
    fibs = [0, 1]                                                                                           
    for f in range(1, k):   
        j=fibs[1]
        fibs[1]=fibs[0]+fibs[1]
        fibs[0]=j                                                                        
    print(fibs[1] / fibs[0])
    return fibs[1] / fibs[0]

def relative_error(k):
    #phi
    phi = (1.0 + 5 ** 0.5) / 2.0
    return abs(r(k) - phi) / phi

n1()
n2(3)
n3()