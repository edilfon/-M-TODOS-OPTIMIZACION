def ingresos_plataforma(suscriptores, ingreso_por_suscriptor, ingreso_adicional):
    ingresos = []
    for suscriptor in suscriptores:
        ingresos.append(ingreso_por_suscriptor * suscriptor + ingreso_adicional)
    return ingresos

# Ejemplo
suscriptores = [1000, 2000, 3000]
ingreso_por_suscriptor = 10
ingreso_adicional = 500
ingresos = ingresos_plataforma(suscriptores, ingreso_por_suscriptor, ingreso_adicional)
print(f"Los ingresos totales son: {ingresos}")
