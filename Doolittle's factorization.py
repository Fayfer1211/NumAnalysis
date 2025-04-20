import numpy as np

def doolittle_factorization(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        # Cálculo de la matriz U
        for k in range(i, n):
            suma = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - suma

        # Cálculo de la matriz L
        L[i][i] = 1  # la diagonal de L siempre es 1 en Doolittle
        for k in range(i+1, n):
            suma = sum(L[k][j] * U[j][i] for j in range(i))
            if U[i][i] == 0:
                raise ValueError("La factorización de Doolittle no es posible: división por cero")
            L[k][i] = (A[k][i] - suma) / U[i][i]
    
    return L, U
