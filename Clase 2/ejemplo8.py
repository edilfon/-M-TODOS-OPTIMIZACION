def energia_consumida(operaciones, energia_por_operacion, energia_base):
    energias = []
    for operacion in operaciones:
        energias.append(energia_por_operacion * operacion + energia_base)
    return energias

# Ejemplo
operaciones = [100, 200, 300]
energia_por_operacion = 0.5
energia_base = 10
energias = energia_consumida(operaciones, energia_por_operacion, energia_base)
print(f"Las energías consumidas totales son: {energias}")
