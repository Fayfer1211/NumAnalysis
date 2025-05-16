import numpy as np


def solve_gc_from_image(A, b, x0=None, M=None, epsilon=1e-6, verbose=False):
    """
    Resuelve el sistema de ecuaciones lineales Ax = b utilizando el algoritmo
    descrito en la imagen (variante del Gradiente Conjugado).

    Parámetros:
    -----------
    A : np.ndarray
        Matriz de coeficientes del sistema (n x n). Debe ser simétrica y definida positiva.
    b : np.ndarray
        Vector de términos independientes (n,).
    x0 : np.ndarray, opcional
        Estimación inicial para la solución x (n,). Si es None, se inicializa con ceros.
    M : int, opcional
        Número máximo de iteraciones. Si es None, se establece a 2*n.
    epsilon : float, opcional
        Tolerancia para la convergencia. El algoritmo se detiene cuando el cuadrado
        de la norma L2 del residuo es menor que epsilon.
    verbose : bool, opcional
        Si es True, imprime información en cada iteración.

    Retorna:
    --------
    x_k : np.ndarray
        La solución aproximada del sistema.
    k : int
        Número de iteraciones realizadas.
    r_norm_sq : float
        El cuadrado de la norma L2 del residuo final.
    """
    n = len(b)

    if A.shape != (n, n):
        raise ValueError(f"La matriz A debe tener dimensiones ({n}, {n}), pero tiene {A.shape}")

    if x0 is None:
        x_k = np.zeros(n, dtype=A.dtype)
    else:
        if x0.shape != (n,):
            raise ValueError(f"x0 debe tener dimensiones ({n},), pero tiene {x0.shape}")
        x_k = x0.astype(A.dtype, copy=True)

    if M is None:
        M = 2 * n

    # Paso 1: r^(0) = b - Ax^(0)
    r_k = b - np.dot(A, x_k)

    # Paso 2: v^(0) = r^(0) (según la corrección común para CG)
    v_k = r_k.copy()

    # Paso 3: output 0, x^(0), r^(0)
    r_k_norm_sq = np.dot(r_k, r_k)
    if verbose:
        print(f"Iteración 0: ||r^(0)||_2^2 = {r_k_norm_sq:.4e}")

    if r_k_norm_sq < epsilon:  # Ya convergido con la estimación inicial
        if verbose:
            print(f"Convergencia en la iteración 0 con la estimación inicial.")
        return x_k, 0, r_k_norm_sq

    for k in range(M):
        # Paso 4: if v^(k) = 0 then stop
        # Chequeamos la norma de v_k. Si es muy pequeña, consideramos que es cero.
        norm_v_k = np.linalg.norm(v_k)
        if norm_v_k < 1e-12:  # Umbral pequeño para considerar v_k como cero
            if verbose:
                print(f"Iteración {k + 1}: Dirección de búsqueda v_k es cercana a cero. Deteniendo.")
            break

        # Cálculo de A*v^(k)
        Av_k = np.dot(A, v_k)

        # Paso 5: t_k = <r^(k), r^(k)> / <v^(k), A*v^(k)>
        # El numerador es r_k_norm_sq (calculado en la iteración anterior o inicial)
        denominador_tk = np.dot(v_k, Av_k)

        if abs(denominador_tk) < 1e-12:  # Evitar división por cero
            if verbose:
                print(f"Iteración {k + 1}: Denominador para t_k es cercano a cero ({denominador_tk:.2e}). Deteniendo.")
                # Esto puede indicar que A no es definida positiva o v_k está en el kernel de A de forma inesperada.
            break

        t_k = r_k_norm_sq / denominador_tk

        # Paso 6: x^(k+1) = x^(k) + t_k * v^(k)
        x_next = x_k + t_k * v_k

        # Paso 7: r^(k+1) = r^(k) - t_k * A*v^(k)
        r_next = r_k - t_k * Av_k

        # Paso 8: if ||r^(k+1)||_2^2 < epsilon then stop
        r_next_norm_sq = np.dot(r_next, r_next)

        if verbose:
            print(f"Iteración {k + 1}: t_k = {t_k:.4e}, ||r^({k + 1})||_2^2 = {r_next_norm_sq:.4e}")

        if r_next_norm_sq < epsilon:
            x_k = x_next
            r_k_norm_sq = r_next_norm_sq
            if verbose:
                print(f"Convergencia alcanzada en la iteración {k + 1}.")
            return x_k, k + 1, r_k_norm_sq

        # Paso 9: s_k = <r^(k+1), r^(k+1)> / <r^(k), r^(k)>
        # El denominador es r_k_norm_sq
        s_k = r_next_norm_sq / r_k_norm_sq

        # Paso 10: v^(k+1) = r^(k+1) + s_k * v^(k)
        v_next = r_next + s_k * v_k

        # Actualizar variables para la siguiente iteración
        x_k = x_next
        r_k = r_next
        v_k = v_next
        r_k_norm_sq = r_next_norm_sq

        # Paso 11: output k+1, x^(k+1), r^(k+1) (manejado por el verbose y el retorno final)

    if verbose:
        if k == M - 1 and r_k_norm_sq >= epsilon:
            print(f"El método no convergió después de {M} iteraciones.")
        elif r_k_norm_sq < epsilon and k < M - 1:  # Convergió en la última iteración posible del bucle
            pass  # El mensaje de convergencia ya se imprimió
        else:  # Se detuvo por denominador_tk o v_k cero
            print(f"El método se detuvo prematuramente en la iteración {k + 1}.")

    return x_k, k + 1, r_k_norm_sq


if __name__ == '__main__':
    # --- Ejemplo 1: Sistema pequeño ---
    print("--- Ejemplo 1: Sistema pequeño ---")
    A1 = np.array([[4., 1.],
                   [1., 3.]])
    b1 = np.array([1., 2.])
    x0_1 = np.array([0., 0.])  # Estimación inicial

    print("Con verbose=True:")
    solucion1, iteraciones1, residuo_sq1 = solve_gc_from_image(A1, b1, x0=x0_1, epsilon=1e-10, verbose=True)
    print(f"\nSolución final (x): {solucion1}")
    print(f"Solución esperada (np.linalg.solve): {np.linalg.solve(A1, b1)}")
    print(f"Iteraciones: {iteraciones1}")
    print(f"Cuadrado de la norma del residuo final: {residuo_sq1:.2e}\n")

    print("Con verbose=False (salida más limpia):")
    solucion1_vF, iteraciones1_vF, residuo_sq1_vF = solve_gc_from_image(A1, b1, x0=x0_1, epsilon=1e-10, verbose=False)
    print(f"Solución final (x): {solucion1_vF}")
    print(f"Iteraciones: {iteraciones1_vF}")
    print(f"Cuadrado de la norma del residuo final: {residuo_sq1_vF:.2e}\n")

    # --- Ejemplo 2: Sistema más grande ---
    print("--- Ejemplo 2: Sistema más grande ---")
    np.random.seed(42)  # Para reproducibilidad
    n_dim = 10
    # Crear una matriz simétrica definida positiva
    L = np.tril(np.random.rand(n_dim, n_dim) * 0.5 + 0.1 * np.eye(n_dim))
    A2 = np.dot(L, L.T) + 0.1 * np.eye(n_dim)
    b2 = np.random.rand(n_dim)
    x0_2 = np.zeros(n_dim)

    solucion2, iteraciones2, residuo_sq2 = solve_gc_from_image(A2, b2, x0=x0_2, epsilon=1e-12, M=100, verbose=True)
    print(f"\nSolución final (primeros 5 elementos de x): {solucion2[:5]}")
    print(f"Solución esperada (primeros 5, np.linalg.solve): {np.linalg.solve(A2, b2)[:5]}")
    print(f"Iteraciones: {iteraciones2}")
    print(f"Cuadrado de la norma del residuo final: {residuo_sq2:.2e}\n")

    # --- Ejemplo 3: Caso donde v_k puede volverse cero (solución exacta en una iteración) ---
    print("--- Ejemplo 3: Solución exacta temprana ---")
    A3 = np.array([[2., 0.],
                   [0., 3.]])
    b3 = np.array([2., 6.])  # Solución exacta es [1, 2]
    x0_3 = np.array([0., 0.])

    solucion3, iteraciones3, residuo_sq3 = solve_gc_from_image(A3, b3, x0=x0_3, epsilon=1e-10, verbose=True)
    print(f"\nSolución final (x): {solucion3}")
    print(f"Iteraciones: {iteraciones3}")
    print(f"Cuadrado de la norma del residuo final: {residuo_sq3:.2e}\n")

    # --- Ejemplo 4: Denominador de t_k podría ser problemático si A no es def. pos. ---
    # (Este algoritmo asume A simétrica y definida positiva)
    # Si A no es definida positiva, el denominador <v, Av> podría ser cero o negativo.
    print("--- Ejemplo 4: Matriz no definida positiva (simulando problema) ---")
    A4_no_def_pos = np.array([[1., 2.],
                              [2., 1.]])  # Eigenvalores 3 y -1
    b4 = np.array([1., 1.])
    try:
        print("Intentando con matriz no definida positiva (puede fallar o no converger bien):")
        solucion4, iteraciones4, residuo_sq4 = solve_gc_from_image(A4_no_def_pos, b4, epsilon=1e-6, verbose=True)
        print(f"\nSolución (x): {solucion4}")
        print(f"Iteraciones: {iteraciones4}")
        print(f"Cuadrado de la norma del residuo final: {residuo_sq4:.2e}")
    except Exception as e:
        print(f"Error como se esperaba o comportamiento no convergente: {e}")

