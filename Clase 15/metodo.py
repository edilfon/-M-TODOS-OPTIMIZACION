import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Regularización Ridge y Lasso", layout="centered")

st.title("Regularización Ridge y Lasso")

# Entrada de datos
st.sidebar.header("Ingrese los datos")
num_puntos = st.sidebar.number_input("Número de datos", min_value=3, max_value=10, value=4, step=1)

X = []
y = []

st.sidebar.subheader("Valores de X y Y")
for i in range(num_puntos):
    col1, col2 = st.sidebar.columns(2)
    x_val = col1.number_input(f"X[{i}]", value=i + 1.0)
    y_val = col2.number_input(f"Y[{i}]", value=i + 2.0)
    X.append([x_val])
    y.append(y_val)

X = np.array(X)
y = np.array(y)

# Parámetros de regularización
st.sidebar.subheader("Parámetro de Regularización (\(\lambda\))")
lambda_ridge = st.sidebar.slider("Lambda Ridge", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
lambda_lasso = st.sidebar.slider("Lambda Lasso", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelos de regresión
ridge = Ridge(alpha=lambda_ridge)
ridge.fit(X_scaled, y)

lasso = Lasso(alpha=lambda_lasso)
lasso.fit(X_scaled, y)

# Resultados
st.header("Resultados")

coef_ridge = ridge.coef_
intercept_ridge = ridge.intercept_

coef_lasso = lasso.coef_
intercept_lasso = lasso.intercept_

# Mostrar tabla de coeficientes
df_resultados = pd.DataFrame({
    "Modelo": ["Ridge Regression", "Lasso Regression"],
    "Coeficientes": [coef_ridge[0], coef_lasso[0]],
    "Intercepto": [intercept_ridge, intercept_lasso]
})

st.table(df_resultados)

# Graficar los modelos
st.subheader("Gráfico de Regresión")

x_plot = np.linspace(min(X_scaled), max(X_scaled), 100)
y_ridge = ridge.predict(x_plot)
y_lasso = lasso.predict(x_plot)

plt.figure(figsize=(8, 5))
plt.scatter(X_scaled, y, color="blue", label="Datos originales")
plt.plot(x_plot, y_ridge, color="red", linestyle="dashed", label="Ridge Regression")
plt.plot(x_plot, y_lasso, color="green", linestyle="dashed", label="Lasso Regression")
plt.xlabel("X (Escalado)")
plt.ylabel("Y")
plt.title("Comparación de Ridge y Lasso")
plt.legend()
st.pyplot(plt)
