def costo_almacenamiento(datos, costo_por_gb, tarifa_fija):
    costos = []
    for dato in datos:
        costos.append(costo_por_gb * dato + tarifa_fija)
    return costos

# Ejemplo
datos = [100, 150, 200]
costo_por_gb = 0.2
tarifa_fija = 10
costos = costo_almacenamiento(datos, costo_por_gb, tarifa_fija)
print(f"Los costos totales de almacenamiento son: {costos}")
