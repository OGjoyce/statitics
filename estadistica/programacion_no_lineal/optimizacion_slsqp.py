#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:44:04 2020

@author: prengbiba
"""

import csv
import numpy as np
from scipy.optimize import minimize
from scipy.stats import gmean
import matplotlib.pyplot as plt

file = open('Data2.csv')
reader = csv.reader(file)
header = next(reader)    #Encabezado y nombre de colunas,
print(header)
#[1990, 32.32, 343.44, 454.55]
dataset = [] #Arreglo vacio.
for row in reader:
    year = int(row[0])
    stocks = float(row[1])
    bounds = float(row[2])
    mm = float(row[3])    
    dataset.append([year, stocks, bounds, mm])


#print(dataset)
    
row0 = dataset[0]
rendStock_array = []
rendBound_array = []
rendMM_array = []

for i in range(1, len(dataset)):
    rendStock = (dataset[i][1] - row0[1])/row0[1]
    rendBound = (dataset[i][2] - row0[2])/row0[2]
    rendMM = (dataset[i][3] - row0[3])/row0[3]
    rendStock_array.append(rendStock)    
    rendBound_array.append(rendBound)    
    rendMM_array.append(rendMM)
    row0 = dataset[i]

#print(rendBound_array)


rendStock_array = np.array(rendStock_array)
rendBound_array = np.array(rendBound_array)
rendMM_array = np.array(rendMM_array)

#Medias Gemoetricas
npArrayStock = 1 + rendStock_array
meanStock = gmean(npArrayStock) - 1 

npArrayBound = 1 + rendBound_array 
meanBound = gmean(npArrayBound) - 1

npArrayMM = 1 +rendMM_array 
meanMM = gmean(npArrayMM) - 1

investment_matrix = np.array([rendStock_array, rendBound_array, rendMM_array])

sigma = np.cov(investment_matrix)
#print(sigma)

mu = np.array([np.mean(rendStock_array), 
               np.mean(rendBound_array), np.mean(rendMM_array)])

#mu = np.array([meanStock, meanBound, meanMM])

#Problema de optimización-----------------------------------------------------#
R = 0.065
#Funciion objetivo
#revisar si se puede con notación de matrices en numpy
def objetivo(x):
    xStock = x[0]
    xBound = x[1]
    xMM = x[2]
    aux1 = xStock*sigma[0,0] + xBound*sigma[0,1] + xMM*sigma[0,2]
    aux2 = xStock*sigma[1,0] + xBound*sigma[1,1] + xMM*sigma[1,2]
    aux3 = xStock*sigma[2,0] + xBound*sigma[2,1] + xMM*sigma[2,2]
    return aux1*xStock + aux2*xBound + aux3*xMM


#Restruccción 1    
def restriccion_1(x):
    return (mu[0]*x[0] + mu[1]*x[1] + mu[2]*x[2] - R)  
    
#Restriccion 2
def restriccion_2(x):
    return (x[0] + x[1] + x[2] - 1)


#valor inicial
x0 = [0,0,0]

rec1 = {'type':'ineq', 'fun':restriccion_1}
rec2 = {'type':'eq' , 'fun': restriccion_2} 

restrs = [rec1, rec2]

solucion = minimize(objetivo, x0, method='SLSQP', constraints=restrs, options={'R':R})

solucion.fun
solucion.x

Solucion_arr = []
R_arr = [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115]
#R_arr = [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.10]
for i in R_arr:
    R = i
    solucion = minimize(objetivo, x0, method='SLSQP', constraints=restrs, options={'R':R})
    Solucion_arr.append(solucion.fun)
    
#graficamos frontera eficiente
plt.plot(Solucion_arr, R_arr, 'ro', linestyle='solid')
plt.xlabel("Riesgo", fontsize=14)
plt.ylabel("Rendimiento", fontsize=14)
plt.suptitle("Frontera Eficiente", fontsize=20)