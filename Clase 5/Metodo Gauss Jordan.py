import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

def gauss_jordan():
    try:
        # Leer la matriz desde la entrada
        matriz = input_text.get("1.0", tk.END).strip()
        matriz = [list(map(float, row.split())) for row in matriz.splitlines()]
        matriz = np.array(matriz)

        if matriz.shape[0] + 1 != matriz.shape[1]:
            messagebox.showerror("Error", "La matriz debe ser una matriz aumentada con n filas y n+1 columnas.")
            return

        pasos = []
        n = len(matriz)

        for i in range(n):
            # Hacer 1 el elemento diagonal
            diag_element = matriz[i, i]
            if diag_element == 0:
                messagebox.showerror("Error", "No se puede resolver: pivote cero encontrado.")
                return
            matriz[i] = matriz[i] / diag_element
            pasos.append(f"Hacer 1 el elemento ({i+1},{i+1}):\n{matriz}")

            # Hacer 0 los elementos por encima y por debajo del pivote
            for j in range(n):
                if i != j:
                    factor = matriz[j, i]
                    matriz[j] = matriz[j] - factor * matriz[i]
                    pasos.append(f"Hacer cero el elemento ({j+1},{i+1}):\n{matriz}")

        soluciones = matriz[:, -1]
        pasos.append(f"Soluciones: {soluciones}")

        output_text.delete("1.0", tk.END)
        for paso in pasos:
            output_text.insert(tk.END, paso + "\n\n")
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error: {e}")

# Crear ventana principal
root = tk.Tk()
root.title("Método Gauss-Jordan")
root.geometry("800x600")

# Estilo de colores
root.configure(bg="#2e8c76")

# Etiqueta de instrucciones
instructions = tk.Label(root, text="Ingrese la matriz aumentada (separe los elementos con espacios y las filas con un salto de línea):", bg="#2e8c76", fg="white", font=("Arial", 12))
instructions.pack(pady=10)

# Cuadro de texto para ingresar la matriz
input_text = tk.Text(root, height=10, width=80, font=("Courier New", 12))
input_text.pack(pady=10)

# Botón para ejecutar el método
calculate_button = ttk.Button(root, text="Mostrar", command=gauss_jordan)
calculate_button.pack(pady=10)

# Cuadro de texto para mostrar resultados
output_text = tk.Text(root, height=20, width=80, font=("Courier New", 12), state="normal", bg="#f0f0f0")
output_text.pack(pady=10)

# Iniciar la aplicación
root.mainloop()
