import streamlit as st
import numpy as np

# Funciones para resolver sistemas de ecuaciones
def resolver_sustitucion(A, b):
    st.write("### Paso 1: Resolución por el método de sustitución")
    # Fórmulas: x_i = (b_i - sum(A_ij * x_j)) / A_ii para cada i
    st.write("La ecuación general es: x_i = (b_i - Σ A_ij * x_j) / A_ii para cada i")
    
    # Solución paso a paso
    st.write("### Pasos para resolver cada incógnita:")
    st.write("Para cada \( x_i \):")
    
    solucion = np.linalg.solve(A, b)
    
    for i in range(len(b)):
        st.write(f"**Ecuación {i+1}:** x_{i+1} = {solucion[i]:.4f}")
        # Mostrar la ecuación que se usó para obtener el valor
        st.write(f"Ecuación para \( x_{i+1} \): x_{i+1} = ({b[i]} - sum(A_{i,:} * x)) / A_{i,i}")
    
    return solucion

def resolver_gauss_jordan(A, b):
    st.write("### Paso 1: Resolución por el método de Gauss-Jordan")
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])
    n = A.shape[0]
    
    st.write("Matriz aumentada inicial:")
    st.write(augmented_matrix)
    st.write("La forma general de Gauss-Jordan es:")
    st.write("Aumentada: [ A | b ]")
    
    for i in range(n):
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]  # Normalizar la fila
        for j in range(n):
            if i != j:
                augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j, i]
        
        # Mostrar los pasos de la matriz aumentada
        st.write(f"Paso {i+1}: Eliminar elementos debajo y arriba de A[{i+1},{i+1}]")
        st.write(augmented_matrix)
    
    st.write("### Resultado:")
    st.write(f"Soluciones: {augmented_matrix[:, -1]}")
    return augmented_matrix[:, -1]

def resolver_cramer(A, b):
    st.write("### Paso 1: Resolución por el método de Cramer")
    det_A = np.linalg.det(A)
    if det_A == 0:
        st.error("El sistema no tiene solución única.")
        return
    st.write(f"Determinante de A: |A| = {det_A}")
    st.write("La fórmula de Cramer es: x_i = |A_i| / |A|, donde A_i es la matriz A reemplazada por la columna b.")
    
    soluciones = []
    for i in range(A.shape[1]):
        Ai = A.copy()
        Ai[:, i] = b
        det_Ai = np.linalg.det(Ai)
        st.write(f"Determinante de A_{i+1} (reemplazando columna {i+1} con b): |A_{i+1}| = {det_Ai}")
        soluciones.append(det_Ai / det_A)
        st.write(f"x_{i+1} = {det_Ai / det_A:.4f}")
    
    return soluciones

# Interfaz de usuario
st.set_page_config(page_title="Resolución de Sistemas de Ecuaciones Lineales", layout="wide")
st.title("🔢 Resolución de Sistemas de Ecuaciones Lineales")

# Barra lateral para ingresar datos
st.sidebar.header("Configuración del Sistema")
n = st.sidebar.number_input("Número de ecuaciones (y variables):", min_value=2, max_value=5, value=3)

# Selección de método
metodo = st.sidebar.selectbox("Selecciona el método:", ["Sustitución", "Gauss-Jordan", "Cramer"])

# Crear las matrices
st.sidebar.header("Entrada de la Matriz y Vector de Términos Independientes")

# Entradas dinámicas para las matrices
A = np.zeros((n, n))
b = np.zeros(n)

# Crear campos para la matriz A
for i in range(n):
    for j in range(n):
        A[i, j] = st.sidebar.number_input(f"a[{i+1},{j+1}]", value=0.0)

# Crear campo para el vector b
for i in range(n):
    b[i] = st.sidebar.number_input(f"b[{i+1}]", value=0.0)

# Mostrar la matriz A y el vector b
st.subheader("Matriz de Coeficientes (A) y Vector de Términos Independientes (B)")
st.write("Matriz A:")
st.write(A)
st.write("Vector B:")
st.write(b)

# Resolución del sistema
if st.sidebar.button("Resolver"):
    if metodo == "Sustitución":
        resultado = resolver_sustitucion(A, b)
    elif metodo == "Gauss-Jordan":
        resultado = resolver_gauss_jordan(A, b)
    elif metodo == "Cramer":
        resultado = resolver_cramer(A, b)
    
    # Mostrar resultados
    st.subheader("Resultados Finales:")
    if isinstance(resultado, str):  # Si el resultado es un mensaje de error
        st.error(resultado)
    else:
        for i, res in enumerate(resultado):
            st.write(f"x{i+1} = {res:.4f}")
