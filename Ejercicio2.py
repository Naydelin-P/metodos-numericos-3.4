#   Codigo que implementa el esquema numerico
#   del metodo iterativo de Gauss-Seidel para
#   resolver sistemas de ecuaciones
#   Naydelin Palomo Martinez
import numpy as np
import matplotlib.pyplot as plt

def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    # Valor incial inicia en 0 por default
    x = np.zeros(n)
    x_prev = np.copy(x)
    errors = []

    for k in range(max_iter):
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_prev[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum1 - sum2) / A[i][i]

        abs_error = np.linalg.norm(x - x_prev, ord=np.inf)
        rel_error = abs_error / (np.linalg.norm(x, ord=np.inf) + 1e-10)
        quad_error = np.linalg.norm(x - x_prev) ** 2

        errors.append((k, abs_error, rel_error, quad_error))

        if abs_error < tol:
            break

        x_prev = np.copy(x)

    return x, errors

A = np.array([
    [20, -5, -3, 0, 0],
    [-4, 18, -2, -1, 0],
    [-3, -1, 22, -5, 0],
    [0, -2, -4, 25, -1],
    [0, 0, 0, -1, 25]])

b = np.array([100, 120, 130, 150, 0])

# Llama a funcion de Gauss-Seidel
x_sol, errors = gauss_seidel(A, b)

# Imprimir errores en formato de tabla
print(f"{'Iteración':<12}{'Error absoluto':<20}{'Error relativo':<20}{'Error cuadrático':<20}")
print("-" * 72)
for k, abs_error, rel_error, quad_error in errors:
    print(f"{k:<12}{abs_error:<20.10e}{rel_error:<20.10e}{quad_error:<20.10e}")

# Imprimir la solución aproximada
print("\nSolución aproximada:")
for i, val in enumerate(x_sol, 1):
    print(f"T{i} = {val:.10e}")


# Graficar errores
iterations = [e[0] for e in errors]
abs_errors = [e[1] for e in errors]
rel_errors = [e[2] for e in errors]
quad_errors = [e[3] for e in errors]

plt.figure(figsize=(10, 5))
plt.plot(iterations, abs_errors, label="Error absoluto")
plt.plot(iterations, rel_errors, label="Error relativo")
plt.plot(iterations, quad_errors, label="Error cuadrático")
plt.yscale("log")
plt.xlabel("Iteraciones")
plt.ylabel("Errores")
plt.title("Convergencia del método de Gauss-Seidel")
plt.legend()
plt.grid()
plt.savefig("convergencia_gauss_seidel.png")
plt.show()
