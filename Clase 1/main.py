import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showerror

def graficar_funcion():
    try:
        # Obtener los valores de los campos de entrada
        funcion = entrada_funcion.get()
        variable = entrada_variable.get()
        min_intervalo = float(entrada_min_intervalo.get())
        max_intervalo = float(entrada_max_intervalo.get())

        # Crear la función simbólica y convertirla en función numérica
        x = symbols(variable)
        funcion_simbolica = eval(funcion)
        funcion_numerica = lambdify(x, funcion_simbolica, "numpy")

        # Crear el intervalo de valores
        x_vals = np.linspace(min_intervalo, max_intervalo, 500)
        y_vals = funcion_numerica(x_vals)

        # Graficar
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label=f"f({variable}) = {funcion}")
        plt.title(f"Gráfico de f({variable}) = {funcion}", fontsize=14)
        plt.xlabel(variable, fontsize=12)
        plt.ylabel(f"f({variable})", fontsize=12)
        plt.axhline(0, color='black', linewidth=0.8, linestyle="--")
        plt.axvline(0, color='black', linewidth=0.8, linestyle="--")
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        showerror("Error", f"Ha ocurrido un error: {e}")

# Crear la interfaz gráfica
ventana = tk.Tk()
ventana.title("Graficador de Funciones")
ventana.geometry("500x400")
ventana.resizable(False, False)

# Configuración de colores
ventana.configure(bg="#2e8c76")  # Fondo de la ventana
estilo = ttk.Style()
estilo.configure("TLabel", background="#2e8c76", foreground="white")  # Colores de las etiquetas
estilo.configure("TButton", background="#630624", foreground="white", padding=5)

# Sección de encabezado
encabezado = tk.Label(
    ventana,
    text=(
        "Universidad Nacional del Altiplano Puno\n"
        "Tema: Variables, funciones y Restricciones\n"
        "Estudiante: Edilfonso Muñoz Anccori"
    ),
    bg="#2e8c76",
    fg="white",
    font=("Arial", 12, "bold"),
    justify="center"
)
encabezado.grid(row=0, column=0, columnspan=2, pady=10)

# Etiquetas y campos de entrada
ttk.Label(ventana, text="Función (f(x))").grid(row=1, column=0, padx=10, pady=10, sticky="w")
entrada_funcion = ttk.Entry(ventana, width=30)
entrada_funcion.grid(row=1, column=1, padx=10, pady=10)

ttk.Label(ventana, text="Variable independiente").grid(row=2, column=0, padx=10, pady=10, sticky="w")
entrada_variable = ttk.Entry(ventana, width=10)
entrada_variable.grid(row=2, column=1, padx=10, pady=10)

ttk.Label(ventana, text="Límite inferior").grid(row=3, column=0, padx=10, pady=10, sticky="w")
entrada_min_intervalo = ttk.Entry(ventana, width=10)
entrada_min_intervalo.grid(row=3, column=1, padx=10, pady=10)

ttk.Label(ventana, text="Límite superior").grid(row=4, column=0, padx=10, pady=10, sticky="w")
entrada_max_intervalo = ttk.Entry(ventana, width=10)
entrada_max_intervalo.grid(row=4, column=1, padx=10, pady=10)

# Botón para graficar
boton_graficar = ttk.Button(ventana, text="Graficar", command=graficar_funcion)
boton_graficar.grid(row=5, column=0, columnspan=2, pady=20)

ventana.mainloop()
