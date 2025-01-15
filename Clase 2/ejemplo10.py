def costo_entrenamiento(iteraciones, costo_por_iteracion, costo_inicial):
    costos = []
    for iteracion in iteraciones:
        costos.append(costo_por_iteracion * iteracion + costo_inicial)
    return costos

# Ejemplo
iteraciones = [500, 1000, 1500]
costo_por_iteracion = 0.1
costo_inicial = 50
costos = costo_entrenamiento(iteraciones, costo_por_iteracion, costo_inicial)
print(f"Los costos totales de entrenamiento son: {costos}")
