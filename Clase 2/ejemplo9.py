def numero_likes(seguidores, proporcion_interaccion, nivel_base):
    likes = []
    for seguidor in seguidores:
        likes.append(proporcion_interaccion * seguidor + nivel_base)
    return likes

# Ejemplo
seguidores = [10000, 15000, 20000]
proporcion_interaccion = 0.02
nivel_base = 50
likes = numero_likes(seguidores, proporcion_interaccion, nivel_base)
print(f"El número total de likes es: {likes}")
