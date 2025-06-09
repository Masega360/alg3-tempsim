import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve

# Construcción matriz A para método implícito
def construir_matriz(nx, ny, dx, dy, dt, alpha):
    Nix, Niy = nx - 2, ny - 2
    Ix = identity(Nix)
    Iy = identity(Niy)

    main_diag_x = 2 * (1/dx**2 + 1/dy**2) * np.ones(Nix)
    off_diag_x = -1/dx**2 * np.ones(Nix - 1)
    Tx = diags([off_diag_x, main_diag_x, off_diag_x], [-1, 0, 1])

    off_diag_y = -1/dy**2 * np.ones(Niy - 1)
    Ty = diags([off_diag_y, off_diag_y], [-1, 1], shape=(Niy, Niy))

    L = kron(Iy, Tx) + kron(Ty, Ix)
    A = identity(Nix*Niy) - dt * alpha * L
    return A

# Inicializar temperatura y condiciones de frontera
def inicializar_T(nx, ny):
    T = np.ones((ny, nx)) * 25
    T[:, 0] = 100    # borde izquierdo
    T[:, -1] = 50    # borde derecho
    T[0, :] = 0      # borde superior
    T[-1, :] = 75    # borde inferior
    # Fuente interna caliente
    T[ny//2 - 1:ny//2 + 2, nx//2 - 1:nx//2 + 2] = 200
    return T

# Método de resolución "optimizado"
import numpy as np

def optimizado(A, d):
    A = A.toarray()  # Convertir a matriz densa
    n = len(d)

    # Extraemos las diagonales a_i (subdiagonal), b_i (principal), c_i (superdiagonal)
    a = np.diag(A, k=-1)  # subdiagonal (a_2 ... a_n), a_1 = 0 implícito
    b = np.diag(A)        # diagonal principal (b_1 ... b_n)
    c = np.diag(A, k=1)   # superdiagonal (c_1 ... c_{n-1}), c_n = 0 implícito

    # Inicializamos los vectores modificados (n elementos)
    b_prime = np.zeros(n)
    d_prime = np.zeros(n)

    # Condiciones iniciales (i=1)
    b_prime[0] = b[0]
    d_prime[0] = d[0]

    # Eliminación hacia adelante: i = 2,...,n
    for i in range(1, n):
        w = a[i-1] / b_prime[i-1]       # coeficiente multiplicador
        b_prime[i] = b[i] - w * c[i-1]
        d_prime[i] = d[i] - w * d_prime[i-1]

    # Sustitución hacia atrás
    x = np.zeros(n)
    x[-1] = d_prime[-1] / b_prime[-1]

    for i in reversed(range(n-1)):
        x[i] = (d_prime[i] - c[i] * x[i+1]) / b_prime[i]

    return x



# Método de Gauss con pivoteo
def gauss_pivoteo(A, b):
    # Convertir a matriz densa para simplicidad (solo para problemas pequeños)
    A = A.toarray().copy()
    b = b.copy()
    n = len(b)

    for k in range(n - 1):
        # Pivoteo parcial: buscar el mayor valor absoluto en la columna k desde fila k hacia abajo
        max_row = np.argmax(np.abs(A[k:, k])) + k
        if abs(A[max_row, k]) < 1e-15:
            raise ValueError("Matriz singular o casi singular")

        # Intercambiar filas si es necesario
        if max_row != k:
            A[[k, max_row], :] = A[[max_row, k], :]
            b[[k, max_row]] = b[[max_row, k]]

        # Eliminación hacia adelante
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in reversed(range(n)):
        if abs(A[i, i]) < 1e-15:
            raise ValueError("Matriz singular o casi singular en sustitución hacia atrás")
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x


# Un paso de simulación
def paso_simulacion(T, A, nx, ny, dx, dy, dt, alpha, metodo_solucion):
    b = T[1:-1, 1:-1].copy()
    b[:, 0] += dt * alpha * T[1:-1, 0] / dx**2
    b[:, -1] += dt * alpha * T[1:-1, -1] / dx**2
    b[0, :] += dt * alpha * T[0, 1:-1] / dy**2
    b[-1, :] += dt * alpha * T[-1, 1:-1] / dy**2
    b = b.flatten()

    if metodo_solucion == 'directo':
        T_vec = spsolve(A, b)
    elif metodo_solucion == 'optimizado':
        T_vec = optimizado(A, b)
    elif metodo_solucion == 'gauss_pivoteo':
        T_vec  = gauss_pivoteo(A, b)
    else:
        raise ValueError("Método de solución no reconocido")

    T_new = T.copy()
    T_new[1:-1, 1:-1] = T_vec.reshape((ny - 2, nx - 2))
    T_new[ny//2 - 1:ny//2 + 2, nx//2 - 1:nx//2 + 2] = 200
    return T_new

# Simulación completa
def simular(nx, ny, dt, alpha, pasos, metodo_solucion):
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    A = construir_matriz(nx, ny, dx, dy, dt, alpha)
    T = inicializar_T(nx, ny)

    tiempos = []
    for _ in range(pasos):
        start = time.time()
        T = paso_simulacion(T, A, nx, ny, dx, dy, dt, alpha, metodo_solucion)
        end = time.time()
        tiempos.append(end - start)

    tiempo_promedio = np.mean(tiempos)
    return T, tiempo_promedio

#----------------------------------------------

import numba
import numpy as np

@numba.njit
def gauss_pivoteo_numba(A, b):
    n = len(b)
    for k in range(n - 1):
        max_row = k
        max_val = abs(A[k, k])
        for r in range(k+1, n):
            if abs(A[r, k]) > max_val:
                max_val = abs(A[r, k])
                max_row = r
        if max_row != k:
            for c in range(k, n):
                A[k, c], A[max_row, c] = A[max_row, c], A[k, c]
            b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= factor * A[k, j]
            b[i] -= factor * b[k]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        suma = 0.0
        for j in range(i+1, n):
            suma += A[i, j] * x[j]
        x[i] = (b[i] - suma) / A[i, i]
    return x


# ---------- Experimentos ----------
resoluciones = [10, 20, 30, 50, 70]
dt = 0.1
alpha = 0.01
pasos = 10
metodos = ['gauss_pivoteo', 'optimizado', 'directo']

for nx in resoluciones:
    ny = nx
    print(f"\nResolución: {nx}x{ny}")
    for metodo in metodos:
        try:
            T_final, tiempo = simular(nx, ny, dt, alpha, pasos, metodo)
            print(f"  Método: {metodo:<15} - Tiempo promedio: {tiempo:.5f} s")
        except Exception as e:
            print(f"  Método: {metodo:<15} - Error: {str(e)}")
