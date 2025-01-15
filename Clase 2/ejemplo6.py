def tiempo_respuesta(solicitudes, tiempo_por_solicitud, tiempo_base):
    tiempos = []
    for solicitud in solicitudes:
        tiempos.append(tiempo_por_solicitud * solicitud + tiempo_base)
    return tiempos

# Ejemplo
solicitudes = [50, 100, 150]
tiempo_por_solicitud = 0.05
tiempo_base = 1
tiempos = tiempo_respuesta(solicitudes, tiempo_por_solicitud, tiempo_base)
print(f"Los tiempos de respuesta promedio son: {tiempos}")
