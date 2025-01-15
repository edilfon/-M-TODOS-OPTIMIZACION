def precio_vivienda(areas, costo_por_m2, costos_fijos):
    precios = []
    for area in areas:
        precios.append(costo_por_m2 * area + costos_fijos)
    return precios

# Ejemplo
areas = [120, 150, 180]
costo_por_m2 = 1000
costos_fijos = 50000
precios = precio_vivienda(areas, costo_por_m2, costos_fijos)
print(f"Los precios de las viviendas son: {precios}")
