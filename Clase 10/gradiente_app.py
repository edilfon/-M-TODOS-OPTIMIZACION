import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la app
st.title("Método de Descenso del Gradiente")
st.write("Simulación del método de descenso del gradiente en una función cuadrática.")

# Función a minimizar: f(x) = ax^2 + bx + c
st.sidebar.header("Parámetros de la función")
a = st.sidebar.slider("Coeficiente a", 0.1, 5.0, 1.0, step=0.1)
b = st.sidebar.slider("Coeficiente b", -10.0, 10.0, 0.0, step=0.1)
c = st.sidebar.slider("Coeficiente c", -10.0, 10.0, 0.0, step=0.1)

# Parámetros del algoritmo
st.sidebar.header("Parámetros del algoritmo")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.01, 1.0, 0.1, step=0.01)
iterations = st.sidebar.slider("Iteraciones máximas", 10, 500, 100, step=10)
x0 = st.sidebar.slider("Valor inicial (x0)", -10.0, 10.0, 0.0, step=0.1)

# Definición de la función y su derivada
def f(x):
    return a * x**2 + b * x + c

def grad_f(x):
    return 2 * a * x + b

# Implementación del descenso del gradiente
x = x0
history = [x]

for _ in range(iterations):
    grad = grad_f(x)
    x = x - learning_rate * grad
    history.append(x)
    # Detener si el gradiente es muy pequeño
    if abs(grad) < 1e-6:
        break

# Crear gráfico
x_vals = np.linspace(min(history) - 1, max(history) + 1, 500)
y_vals = f(x_vals)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_vals, y_vals, label="f(x)")
ax.scatter(history, [f(x) for x in history], color="red", label="Iteraciones", zorder=5)
ax.plot(history, [f(x) for x in history], color="red", linestyle="--", alpha=0.7)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Método de Descenso del Gradiente")
ax.legend()
ax.grid()

# Mostrar resultados
st.pyplot(fig)
st.write("### Resultados:")
st.write(f"Valor inicial: {x0}")
st.write(f"Valor mínimo encontrado: {x:.4f}")
st.write(f"f(x) mínimo: {f(x):.4f}")
st.write(f"Iteraciones realizadas: {len(history) - 1}")
