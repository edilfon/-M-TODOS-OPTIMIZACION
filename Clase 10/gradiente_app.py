import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Configuración de la app
st.title("Método de Descenso del Gradiente")
st.write("Simulación del método de descenso del gradiente para minimizar una función simbólica.")

# Entrada de la ecuación desde el teclado
st.sidebar.header("Entrada de la función")
equation = st.sidebar.text_input("Ingresa la función f(x):", value="x**2 - 4*x + 4")
x_symbol = sp.Symbol("x")  # Variable simbólica
try:
    f_expr = sp.sympify(equation)  # Convierte la ecuación a simbólica
    f_derivative = sp.diff(f_expr, x_symbol)  # Calcula la derivada simbólica
except sp.SympifyError:
    st.sidebar.error("La función ingresada no es válida. Usa solo 'x' como variable.")
    st.stop()

# Parámetros del algoritmo
st.sidebar.header("Parámetros del algoritmo")
learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.01, 1.0, 0.1, step=0.01)
iterations = st.sidebar.slider("Iteraciones máximas", 10, 500, 100, step=10)
x0 = st.sidebar.slider("Valor inicial (x0)", -10.0, 10.0, 0.0, step=0.1)

# Conversión de funciones simbólicas a funciones numéricas
f = sp.lambdify(x_symbol, f_expr, "numpy")
grad_f = sp.lambdify(x_symbol, f_derivative, "numpy")

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

# Crear el gráfico
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
st.write(f"Función ingresada: {equation}")
st.write(f"Derivada calculada: {f_derivative}")
st.write(f"Valor inicial: {x0}")
st.write(f"Valor mínimo encontrado: {x:.4f}")
st.write(f"f(x) mínimo: {f(x):.4f}")
st.write(f"Iteraciones realizadas: {len(history) - 1}")
