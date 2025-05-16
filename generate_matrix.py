import random

def generar_matriz_texto(filas, columnas, valor_min=0, valor_max=9):
    """
    Genera una matriz de dimensiones filas x columnas con números aleatorios
    y la devuelve como una cadena de texto formateada.
    """
    matriz_texto_lista = []
    for i in range(filas):
        fila_numeros = [str(random.randint(valor_min, valor_max)) for _ in range(columnas)]
        matriz_texto_lista.append(" ".join(fila_numeros))  # Une los números de la fila con espacios
    return "\n".join(matriz_texto_lista) # Une las filas con saltos de línea

def main():
    print("Generador de Matriz Numérica")

    while True:
        try:
            filas = int(input("Introduce el número de filas para la matriz: "))
            if filas <= 0:
                print("El número de filas debe ser un entero positivo.")
                continue
            break
        except ValueError:
            print("Entrada no válida. Por favor, introduce un número entero para las filas.")

    while True:
        try:
            columnas = int(input("Introduce el número de columnas para la matriz: "))
            if columnas <= 0:
                print("El número de columnas debe ser un entero positivo.")
                continue
            break
        except ValueError:
            print("Entrada no válida. Por favor, introduce un número entero para las columnas.")

    # Generar la matriz como texto
    matriz_generada_texto = generar_matriz_texto(filas, columnas)

    # Mostrar la matriz generada en la consola
    print("\n--- Matriz Generada ({0}x{1}) ---".format(filas, columnas))
    print(matriz_generada_texto)
    print("--- Fin de la Matriz ---")

    # Preguntar si se desea guardar la matriz en un archivo
    guardar = input("\n¿Deseas guardar esta matriz en un archivo de texto? (s/N): ").strip().lower()
    if guardar == 's':
        nombre_archivo_sugerido = "matriz_{0}x{1}.txt".format(filas, columnas)
        nombre_archivo = input(f"Introduce el nombre del archivo (por defecto: {nombre_archivo_sugerido}): ")
        if not nombre_archivo:
            nombre_archivo = nombre_archivo_sugerido

        try:
            with open(nombre_archivo, "w") as archivo:
                archivo.write(matriz_generada_texto)
            print(f"Matriz guardada en '{nombre_archivo}'")
        except IOError:
            print(f"Error: No se pudo guardar el archivo '{nombre_archivo}'.")

if __name__ == "__main__":
    main()