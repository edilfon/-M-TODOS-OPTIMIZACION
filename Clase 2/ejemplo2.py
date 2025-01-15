def ganancia_mensual(predicciones, ganancia_por_prediccion, ingresos_fijos):
    ganancias = []
    for prediccion in predicciones:
        ganancias.append(ganancia_por_prediccion * prediccion + ingresos_fijos)
    return ganancias

# Ejemplo
predicciones = [200, 300, 400]
ganancia_por_prediccion = 50
ingresos_fijos = 1000
ganancias = ganancia_mensual(predicciones, ganancia_por_prediccion, ingresos_fijos)
print(f"Las ganancias mensuales son: {ganancias}")
