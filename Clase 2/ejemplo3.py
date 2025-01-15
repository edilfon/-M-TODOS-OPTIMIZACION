def tiempo_procesamiento(datos, tiempo_por_dato, tiempo_configuracion):
    tiempos = []
    for dato in datos:
        tiempos.append(tiempo_por_dato * dato + tiempo_configuracion)
    return tiempos

# Ejemplo
datos = [500, 600, 700]
tiempo_por_dato = 0.1
tiempo_configuracion = 5
tiempos = tiempo_procesamiento(datos, tiempo_por_dato, tiempo_configuracion)
print(f"Los tiempos totales de procesamiento son: {tiempos}")
