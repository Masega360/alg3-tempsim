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
    A = A.toarray()
    n = len(d)

    #
    a = np.diag(A, k=-1)
    b = np.diag(A)        # diagonal principal (b_1 ... b_n)
    c = np.diag(A, k=1)


    b_prime = np.zeros(n)
    d_prime = np.zeros(n)


    b_prime[0] = b[0]
    d_prime[0] = d[0]

    # Eliminación hacia adelante: i = 2,...,n
    for i in range(1, n):
        w = a[i-1] / b_prime[i-1]       # m
        b_prime[i] = b[i] - w * c[i-1]
        d_prime[i] = d[i] - w * d_prime[i-1]

    # Sustitución hacia atrás
    x = np.zeros(n)
    x[-1] = d_prime[-1] / b_prime[-1]

    for i in reversed(range(n-1)):
        x[i] = (d_prime[i] - c[i] * x[i+1]) / b_prime[i]

    return x



# Método de Gauss con pivoteo
import numpy as np

def gauss_pivoteo(A, b):
    A = A.toarray().copy()
    b = b.copy()
    n = len(b)

    for k in range(n - 1):
        # Pivoteo parcial: buscar el mayor valor absoluto en la columna k desde fila k hacia abajo
        max_row = np.argmax(np.abs(A[k:, k])) + k
        if abs(A[max_row, k]) < 1e-15:
            raise ValueError("Matriz singular o casi singular")

        if max_row != k:
            A[[k, max_row], :] = A[[max_row, k], :]
            b[[k, max_row]] = b[[max_row, k]]

        pivot = A[k, k]
        A[k, :] = A[k, :] / pivot
        b[k] = b[k] / pivot

        for i in range(k + 1, n):
            factor = A[i, k]
            A[i, :] -= factor * A[k, :]
            b[i] -= factor * b[k]

    if abs(A[n-1, n-1]) < 1e-15:
        raise ValueError("Matriz singular o casi singular")
    b[n-1] = b[n-1] / A[n-1, n-1]
    A[n-1, :] = A[n-1, :] / A[n-1, n-1]

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = b[i] - np.dot(A[i, i + 1:], x[i + 1:])

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



resoluciones = [10, 20, 30, 50, 70]
dt = 0.1
alpha = 0.01
pasos = 10
metodos = ['gauss_pivoteo', 'optimizado', 'directo']

tiempos_por_metodo = {metodo: [] for metodo in metodos}
soluciones_finales = {metodo: [] for metodo in metodos}

for nx in resoluciones:
    ny = nx
    print(f"\nResolución: {nx}x{ny}")
    for metodo in metodos:
        try:
            T_final, tiempo = simular(nx, ny, dt, alpha, pasos, metodo)
            tiempos_por_metodo[metodo].append(tiempo)
            soluciones_finales[metodo].append(T_final)
            print(f"  Método: {metodo:<15} - Tiempo promedio: {tiempo:.5f} s")
        except Exception as e:
            tiempos_por_metodo[metodo].append(np.nan)
            soluciones_finales[metodo].append(None)
            print(f"  Método: {metodo:<15} - Error: {str(e)}")

# ----------- Gráficos de tiempo separados -----------
for metodo in metodos:
    plt.figure()
    plt.plot(resoluciones, tiempos_por_metodo[metodo], marker='o')
    plt.title(f"Tiempo de simulación promedio - Método: {metodo}")
    plt.xlabel("Resolución (nx = ny)")
    plt.ylabel("Tiempo promedio por paso (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------- Cálculo de errores relativos respecto al método 'directo' -----------
errores_relativos = {'gauss_pivoteo': [], 'optimizado': []}

for i, res in enumerate(resoluciones):
    ref = soluciones_finales['directo'][i]
    ref_vec = ref.flatten() if ref is not None else None
    for metodo in ['gauss_pivoteo', 'optimizado']:
        T_metodo = soluciones_finales[metodo][i]
        if ref_vec is not None and T_metodo is not None:
            diff = T_metodo.flatten() - ref_vec
            error_rel = np.linalg.norm(diff) / np.linalg.norm(ref_vec)
        else:
            error_rel = np.nan
        errores_relativos[metodo].append(error_rel)

# ----------- Gráfico de errores relativos -----------
plt.figure(figsize=(10, 5))
for metodo in errores_relativos:
    plt.plot(resoluciones, errores_relativos[metodo], marker='x', label=f"{metodo}")
plt.title("Error relativo respecto al método 'directo'")
plt.xlabel("Resolución (nx = ny)")
plt.ylabel("Error relativo")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ----------- Gráficos individuales de errores relativos (excepto 'directo') -----------
for metodo in errores_relativos:
    plt.figure()
    plt.plot(resoluciones, errores_relativos[metodo], marker='o', color='tab:red')
    plt.title(f"Error relativo del método '{metodo}' respecto al método 'directo'")
    plt.xlabel("Resolución (nx = ny)")
    plt.ylabel("Error relativo")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

