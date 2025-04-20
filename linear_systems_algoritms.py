# -*- coding: utf-8 -*-
import numpy as np

# ------------------------------
# Entrada de matriz A y vector b
# ------------------------------
def input_matrix():
    """
    Solicita al usuario que ingrese una matriz A (n x n) y un vector b de tamaño n.

    Returns:
        A (ndarray): Matriz de coeficientes.
        b (ndarray): Vector del lado derecho del sistema.
    """
    n = int(input("Ingrese el tamaño del sistema (n x n): "))
    print("Ingrese la matriz A (una fila por línea, separada por espacios):")
    A = np.array([list(map(float, input(f"Fila {i+1}: ").split())) for i in range(n)])
    print("Ingrese el vector b (separado por espacios):")
    b = np.array(list(map(float, input().split())))
    return A, b

# ------------------------------
# Validaciones
# ------------------------------
def es_simetrica(A):
    """Verifica si la matriz A es simétrica."""
    return np.allclose(A, A.T)

def es_definida_positiva(A):
    """Verifica si la matriz A es definida positiva usando descomposición de Cholesky."""
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def cumple_criterio_filas(A):
    """Verifica si la matriz A cumple con el criterio de diagonal dominante por filas."""
    for i in range(len(A)):
        if abs(A[i][i]) <= sum(abs(A[i][j]) for j in range(len(A)) if j != i):
            return False
    return True

def cumple_criterio_columnas(A):
    """Verifica si la matriz A cumple con el criterio de diagonal dominante por columnas."""
    for j in range(len(A)):
        if abs(A[j][j]) <= sum(abs(A[i][j]) for i in range(len(A)) if i != j):
            return False
    return True

def cumple_criterio_diagonal_dominante(A):
    """Verifica si la matriz A cumple con el criterio de diagonal dominante por filas o columnas."""
    return cumple_criterio_filas(A) or cumple_criterio_columnas(A)

# ------------------------------
# 1. Eliminación de Gauss
# ------------------------------
def gaussian_elimination(A, b):
    """
    Resuelve el sistema Ax = b usando el método de eliminación de Gauss.

    Args:
        A (ndarray): Matriz de coeficientes.
        b (ndarray): Vector del lado derecho.

    Returns:
        x (ndarray): Solución del sistema.
    """
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i][i]
    return x

# ------------------------------
# 2. Eliminación de Gauss-Jordan
# ------------------------------
def gauss_jordan_elimination(A, b):
    """
    Resuelve el sistema Ax = b usando el método de eliminación de Gauss-Jordan.

    Args:
        A (ndarray): Matriz de coeficientes.
        b (ndarray): Vector del lado derecho.

    Returns:
        x (ndarray): Solución del sistema.
    """
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    for i in range(n):
        A[i] /= A[i][i]
        b[i] /= A[i][i]
        for j in range(n):
            if i != j:
                factor = A[j][i]
                A[j] -= factor * A[i]
                b[j] -= factor * b[i]
    return b

# ------------------------------
# 3. Factorización LU (Doolittle)
# ------------------------------
def doolittle_factorization(A):
    """
    Realiza la factorización LU de A usando el método de Doolittle.

    Args:
        A (ndarray): Matriz de coeficientes cuadrada.

    Returns:
        L (ndarray): Matriz triangular inferior.
        U (ndarray): Matriz triangular superior.
    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            U[i][k] = A[i][k] - sum(L[i][j] * U[j][k] for j in range(i))
        L[i][i] = 1
        for k in range(i + 1, n):
            L[k][i] = (A[k][i] - sum(L[k][j] * U[j][i] for j in range(i))) / U[i][i]
    return L, U

def lu_solve(L, U, b):
    """
    Resuelve el sistema LUx = b mediante sustitución hacia adelante y atrás.

    Args:
        L (ndarray): Matriz triangular inferior.
        U (ndarray): Matriz triangular superior.
        b (ndarray): Vector del lado derecho.

    Returns:
        x (ndarray): Solución del sistema.
    """
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]
    return x

# ------------------------------
# 4. Factorización LU (Crout)
# ------------------------------
def crout_factorization(A):
    """
    Realiza la factorización LU de A usando el método de Crout.

    Args:
        A (ndarray): Matriz de coeficientes cuadrada.

    Returns:
        L (ndarray): Matriz triangular inferior.
        U (ndarray): Matriz triangular superior con diagonal unitaria.
    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.identity(n)

    for j in range(n):
        for i in range(j, n):
            L[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(j))
        for i in range(j + 1, n):
            U[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(j))) / L[j][j]
    return L, U

# ------------------------------
# 5. Factorización de Cholesky
# ------------------------------
def cholesky_factorization(A):
    """
    Realiza la factorización de Cholesky para una matriz simétrica definida positiva.

    Args:
        A (ndarray): Matriz SPD.

    Returns:
        L (ndarray): Matriz triangular inferior tal que A = LL^T.
    """
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            suma = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - suma)
            else:
                L[i][j] = (A[i][j] - suma) / L[j][j]
    return L

def cholesky_solve(L, b):
    """
    Resuelve el sistema Ax = b usando la factorización de Cholesky.

    Args:
        L (ndarray): Matriz triangular inferior.
        b (ndarray): Vector del lado derecho.

    Returns:
        x (ndarray): Solución del sistema.
    """
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)

    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / L[i][i]
    return x

# ------------------------------
# 6. Método de Jacobi
# ------------------------------
def jacobi_method(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Resuelve el sistema Ax = b usando el método iterativo de Jacobi.

    Args:
        A (ndarray): Matriz de coeficientes.
        b (ndarray): Vector del lado derecho.
        x0 (ndarray): Aproximación inicial (opcional).
        tol (float): Tolerancia para el criterio de convergencia.
        max_iter (int): Número máximo de iteraciones.

    Returns:
        x (ndarray): Aproximación a la solución.
    """
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()

    for _ in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# ------------------------------
# 7. Método de Gauss-Seidel
# ------------------------------
def gauss_seidel_method(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Resuelve el sistema Ax = b usando el método iterativo de Gauss-Seidel.

    Args:
        A (ndarray): Matriz de coeficientes.
        b (ndarray): Vector del lado derecho.
        x0 (ndarray): Aproximación inicial (opcional).
        tol (float): Tolerancia para el criterio de convergencia.
        max_iter (int): Número máximo de iteraciones.

    Returns:
        x (ndarray): Aproximación a la solución.
    """
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()

    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# ------------------------------
# 8. Gradiente Conjugado
# ------------------------------
def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    Resuelve el sistema Ax = b usando el método del gradiente conjugado.
    Requiere que A sea simétrica y definida positiva.

    Args:
        A (ndarray): Matriz simétrica definida positiva.
        b (ndarray): Vector del lado derecho.
        x0 (ndarray): Aproximación inicial (opcional).
        tol (float): Tolerancia para el criterio de convergencia.
        max_iter (int): Número máximo de iteraciones.

    Returns:
        x (ndarray): Aproximación a la solución.
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - A @ x
    p = r.copy()

    for _ in range(max_iter):
        Ap = A @ p
        alpha = r @ r / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        if np.linalg.norm(r_new) < tol:
            return x
        beta = r_new @ r_new / (r @ r)
        p = r_new + beta * p
        r = r_new
    return x
