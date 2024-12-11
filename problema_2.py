import matplotlib

# Usamos el backend TkAgg para que funcione en algunos entornos de Windows
matplotlib.use("TkAgg")
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Definimos la función para mostrar imágenes
def imshow(
    img,
    new_fig=True,
    title=None,
    color_img=False,
    blocking=False,
    colorbar=False,
    ticks=False,
):
    # Si new_fig es True, creamos una nueva figura
    if new_fig:
        plt.figure()

    # Mostrar imagen en color o en escala de grises, según el parámetro color_img
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray")

    # Añadir título a la imagen si se proporciona
    plt.title(title)
    
    # Si ticks es False, eliminamos las marcas de los ejes
    if not ticks:
        plt.xticks([]), plt.yticks([])

    # Si colorbar es True, mostramos la barra de colores
    if colorbar:
        plt.colorbar()

    # Mostrar la figura
    if new_fig:
        plt.show(block=blocking)

# Lista para almacenar las imágenes procesadas
imagenes = []

# Lista de nombres de imágenes (asumido que las imágenes están en formato "img01.png", "img02.png", etc.)
for num_foto in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
    print(num_foto)  # Imprime el nombre de la imagen que se está procesando

    # Leer la imagen original y convertirla de BGR (formato OpenCV) a RGB
    patente = cv2.imread(f"img{num_foto}.png", cv2.IMREAD_COLOR)
    patente_original = cv2.cvtColor(patente, cv2.COLOR_BGR2RGB)

    # Convertir la imagen original a escala de grises para procesamiento
    patente_gray = cv2.cvtColor(patente_original, cv2.COLOR_RGB2GRAY)

    # Umbral para hacer iterativo el método de detección de patentes, personalizado para cada patente
    umbral = 49
    bandera = True  # Bandera para controlar la iteración del umbral
    while bandera:
        umbral = umbral + 1  # Incrementar el umbral

        # Si el umbral supera 250, detener el ciclo
        if umbral > 250:
            bandera = False
            break

        # Aplicar el umbral binario
        _, thresh = cv2.threshold(patente_gray, umbral, 255, cv2.THRESH_BINARY)

    # ---------------------------------------------------------------------------------------------------------------------------

        # Obtener los componentes conectados en la imagen umbralizada
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 4)

        l_aspecto = []  # Lista para almacenar las componentes con una relación de aspecto adecuada

        # Filtrar componentes con área adecuada
        filtered_labels = np.zeros_like(labels, dtype=np.uint8)
        for i_label in range(1, num_labels):  # Ignorar el fondo (etiqueta 0)
            area = stats[i_label, cv2.CC_STAT_AREA]
            # Filtrar componentes según su área
            if 17 < area < 200:
                # Calcular la relación de aspecto (alto/ancho)
                relacion_aspecto = stats[i_label][3] / stats[i_label][2]
                # Filtrar por relación de aspecto
                if 1.5 <= relacion_aspecto <= 3.0:
                    l_aspecto.append(i_label)

        # Suponemos que son 6 las componentes de la patente
        if len(l_aspecto) >= 6:
            centro_caja = []  # Lista para almacenar los centros de las cajas de las componentes filtradas
            for z in l_aspecto:
                x, y, w, h, a = stats[z]
                cx2 = x + (w / 2)  # Coordenada x del centro
                cy2 = y + (h / 2)  # Coordenada y del centro
                centro_caja.append([z, cx2, cy2])

            # Comprobar las distancias entre las componentes y agrupar las que están cerca
            for k in range(len(centro_caja)):
                j, cx, cy = centro_caja[k]
                cajas_finales = []  # Lista para almacenar las cajas agrupadas
                cajas_finales.append(j)
                for f in range(len(centro_caja)):
                    if f != k:
                        j3, cx3, cy3 = centro_caja[f]
                        # Calcular las distancias en x e y entre las componentes
                        resta_x = abs(cx - cx3)
                        resta_y = abs(cy - cy3)
                        # Si las componentes están cerca en el eje X y Y, las agrupamos
                        if 0.0 < resta_x < 80.0 and 0.0 < resta_y < 20.0:
                            cajas_finales.append(j3)
                # Si el número de componentes agrupadas está en un rango adecuado, detenemos la iteración
                if len(cajas_finales) >= 6 and len(cajas_finales) < 8:
                    bandera = False
                    break

    # Foto final con las componentes detectadas
    esq_x_menor = 1000  # Inicializar las coordenadas de las esquinas de la caja
    esq_y_menor = 1000
    esq_x_mayor = 0
    esq_y_mayor = 0

    copia_foto_original = patente_original.copy()  # Copia de la imagen original para dibujar las cajas

    # Dibujar las cajas finales detectadas
    for num in cajas_finales:
        x, y, w, h, a = stats[num]
        if a > 10:  # Filtrar cajas con área suficiente
            # Coordenadas del rectángulo
            punto1 = (x, y)  # Esquina superior izquierda
            punto2 = (x + w, y + h)  # Esquina inferior derecha
            # Dibujar el rectángulo verde
            cv2.rectangle(copia_foto_original, punto1, punto2, color=(0, 255, 0), thickness=1)
            # Actualizar las coordenadas de las esquinas
            if x < esq_x_menor:
                esq_x_menor = x
            if y < esq_y_menor:
                esq_y_menor = y
            if x + w > esq_x_mayor:
                esq_x_mayor = x + w
            if y + h > esq_y_mayor:
                esq_y_mayor = y + h

    # Dibujar un rectángulo azul alrededor de la patente detectada
    punto1 = (esq_x_menor - 5, esq_y_menor - 5)  # Esquina superior izquierda
    punto2 = (esq_x_mayor + 5, esq_y_mayor + 5)  # Esquina inferior derecha
    cv2.rectangle(copia_foto_original, punto1, punto2, color=(255, 0, 0), thickness=2)

    # Añadir la imagen procesada a la lista
    imagenes.append(copia_foto_original)

# Organizar las imágenes en una cuadrícula
n_rows = 4  # Número de filas
n_cols = 3  # Número de columnas

# Redimensionar las imágenes para que todas tengan el mismo tamaño
height, width, _ = imagenes[0].shape
imagenes_resized = [cv2.resize(img, (width, height)) for img in imagenes]

# Concatenar las imágenes por filas
imagenes_finales = []
for i in range(n_rows):
    imagen_fila = np.hstack(imagenes_resized[i * n_cols:(i + 1) * n_cols])
    imagenes_finales.append(imagen_fila)

# Concatenar todas las filas en una sola imagen
imagen_completa = np.vstack(imagenes_finales)

# Crear una ventana redimensionable
cv2.namedWindow('Cuadrícula de Imágenes', cv2.WINDOW_NORMAL)

# Ajustar el tamaño de la ventana (por ejemplo, 1080x1920)
cv2.resizeWindow('Cuadrícula de Imágenes', 1080, 1920)

# Mostrar la imagen final con todas las imágenes en la cuadrícula
cv2.imshow('Cuadrícula de Imágenes', imagen_completa)

# Esperar hasta que se presione una tecla para cerrar la ventana
cv2.waitKey(0)
cv2.destroyAllWindows()