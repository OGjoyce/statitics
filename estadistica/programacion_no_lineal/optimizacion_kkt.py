#SCRIPT ELBORADO PARA USO DE ESTUDIO, CON  EL FIN DE PODER COMPRENDER LOS CONCEPTOS ESTUDIADOS
#DURANTE LOS DIAS DE CLASE, ESTE FUE DESARROLLADO POR EL EQUIPO CARLOS MONTIEL, PEPE SOTO Y NEHEMIAS LÓPEZ

import pandas as pd
from sympy import *
import numpy as np
import matplotlib.pyplot as plt

list_stocks = []
list_bonds = []
list_mm = []

#tasa de rendimiento
def Yielrate(P, Pt_1):
    r_i = (P - Pt_1)/Pt_1
    return r_i

def Lists_items(df):
    for i in range(1, len(df)):
        list_stocks.append(Yielrate(df[" stocks"][i], df[" stocks"][i-1]))
        list_bonds.append(Yielrate(df[" bonds"][i], df[" bonds"][i-1]))
        list_mm.append(Yielrate(df[" mm"][i], df[" mm"][i-1]))

    list_rate = {'stocks' : list_stocks, 'bonds' : list_bonds, 'mm' : list_mm}
    return pd.DataFrame(list_rate)

rprm_stocks = 0
rprm_bonds = 0
rprm_mm = 0

#tasa de rendimiento promedio
def Yielratemean(T, rprm_stocks, rprm_bonds, rprm_mm):
    for i in range(len(list_stocks)):
        rprm_stocks = list_stocks[i] + rprm_stocks
        rprm_bonds = list_bonds[i] + rprm_bonds
        rprm_mm = list_mm[i] + rprm_mm
    rprm_stocks = rprm_stocks/T
    rprm_bonds = rprm_bonds/T
    rprm_mm = rprm_mm/T
    return [rprm_stocks, rprm_bonds, rprm_mm]

#Mariz de covarianza
column_j = ""
column_k = ""
sigma = []

def Matrixcovariance(frame_rate):
    result = 0
    rows = []
    for i in range(0,3):
        column_j = frame_rate.columns[i]
        for j in range(0,3):
            column_k = frame_rate.columns[j]
            for k in range(len(list_mm)):
                result += ((frame_rate[column_j][k] - rprm_array[i])*(frame_rate[column_k][k] - rprm_array[j]))
            rows.append(result/43)
            result = 0
        sigma.append(rows)
        rows = []
    return sigma

x,y,z = symbols('x y z')
variables = Matrix([x,y,z])

#Devolvemos la funcion objetivo
def Funcobjective(sigma, variables):
    sigma_convert = Matrix(sigma)
    function = variables.T*sigma_convert*variables
    return function

#Devolvemos restricciones
def Restrictions(mu, variables):
    mu_convert = Matrix(mu)
    restrictions_ineq = mu_convert.T*variables
    return [restrictions_ineq, variables[0] + variables[1] + variables[2] ]

#Devolvemos el gradiente de la funcion
def Gradientfunction(f, variables):
    gradiente_function_array = []
    for i in range(0,3):
        gradiente_function_array.append(f[0].diff(variables[i]))
    return (Matrix(gradiente_function_array))

#Devolvemos el gradiente de la ecuación de desigualdad.
def Gradientinequality(equationsineq, variables):
    gradiente_ineq_array = []
    m = symbols('m')
    for i in range(0,3):
        gradiente_ineq_array.append(equationsineq[0].diff(variables[i]))
    return (Matrix(gradiente_ineq_array).T*m)

#Devolvemos el gradiente de la ecuación de igualdad.
def GradientEquality(equationseq, variables):
    gradiente_eq_array = []
    l = symbols('l')
    for i in range(0,3):
        gradiente_eq_array.append(equationseq.diff(variables[i]))
    return (Matrix(gradiente_eq_array).T*l)

def Minvalue(resolve, sigma, r):
    values_array = []
    for j in range(0, len(resolve)):
        variables = [resolve[j][0], resolve[j][1], resolve[j][2]]
        variables = Matrix([variables])
        values_array.append(variables*Matrix(sigma)*variables.T)
        min_value = np.amin(np.array(values_array))
        print("|            {:^10}            |            {:^10}            |".format(str(r),str(min_value)))
    return min_value

def Minimize(matrixequations, restrictions, sigma):
    x, y, z, l, m, k = symbols('x y z l m k')
    eq_array = []
    solution = []
    for i in range(0, len(matrixequations)):
        eq_array.append(Eq(matrixequations[i],0))
    ecuation = (r[0][0] - 0.05)*m
    eq_array.append(Eq(ecuation,0))
    ecuation = r[1]-1
    eq_array.append(Eq(ecuation,0))
    R_arr = [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115, 0.12, 0.125]
    ecuation = (r[0][0]-0.05)
    eq_array.append(Eq(ecuation,0))
    print(" __________________________________ _________________________________________")
    print("|        {:^10}       |                {:^10}             |".format("portafolios óptimos","valor optimo"))
    print(" ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅  ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ̅ ")
    for rarr in R_arr:
        eq_array[3] = Eq((r[0][0] - rarr)*m,0)
        eq_array[5] = Eq((r[0][0] - rarr),0)
        resolve = solve(eq_array, [x,y,z,l,m,k])
        solution.append(Minvalue(resolve, sigma, rarr))
    return solution, R_arr

def Graph(Solucion_arr, R_arr):
    plt.plot(Solucion_arr, R_arr, 'ro', linestyle='solid')
    plt.xlabel("Riesgo", fontsize=14)
    plt.ylabel("Rendimiento", fontsize = 14)
    plt.suptitle("Frontera eficiente", fontsize=14)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('Data2.csv')
    T = len(df)
    frame_rate = Lists_items(df)
    rprm_array = Yielratemean(T, rprm_stocks, rprm_bonds, rprm_mm)
    sigma = Matrixcovariance(frame_rate)
    f = Funcobjective(sigma, variables)
    r = Restrictions(rprm_array, variables)
    Gf = Gradientfunction(f, variables)
    Gg = Gradientinequality(r[0], variables)
    Gh = GradientEquality(r[1], variables)
    kkT = Gf + Gh.T + Gg.T
    print("MINIMIZANDO....")
    solution, R_arr =  Minimize(kkT, r, sigma)
    print("|__________________________________|_________________________________________|")
    Graph(solution, R_arr)

    