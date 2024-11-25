import matplotlib

matplotlib.use("TkAgg")
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Defininimos función para mostrar imágenes
def imshow(
    img,
    new_fig=True,
    title=None,
    color_img=False,
    blocking=False,
    colorbar=False,
    ticks=False,
):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap="gray")
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


for i in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
    # for i in ['01', '02']:
    patente = cv2.imread(f"img{i}.png", cv2.IMREAD_COLOR)
    patente_original = cv2.cvtColor(patente, cv2.COLOR_BGR2RGB)
    patente_gray = cv2.cvtColor(patente_original, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(patente_gray, 100, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    # Filtrar componentes con área mayor a 300
    filtered_labels = np.zeros_like(labels, dtype=np.uint8)  # Crear una nueva máscara
    for i in range(1, num_labels):  # Ignorar el fondo (etiqueta 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 300 or area > 150000:
            filtered_labels[labels == i] = 255  # Mantener el componente

    # imshow(filtered_labels, title='Original')

    # Encontrar contornos
    contours, _ = cv2.findContours(
        filtered_labels, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Crear una copia de la imagen para dibujar los resultados
    output = cv2.cvtColor(filtered_labels, cv2.COLOR_GRAY2BGR)

    # Iterar sobre cada contorno
    for contour in contours:
        # Calcular el bounding box (rectángulo que encierra el objeto)
        x, y, w, h = cv2.boundingRect(contour)

        # Aplicar el filtro: mantener objetos con relación de aspecto entre 0.5 y 2
        if 40 < w < 100 and 15 < h < 100:
            # Dibujar el contorno que pasa el filtro
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    imshow(output, title="Original")

    # a la imagen 1 lo hace


# Pruebas
patente = cv2.imread("img02.png", cv2.IMREAD_COLOR)
patente_original = cv2.cvtColor(patente, cv2.COLOR_BGR2RGB)
patente_gray = cv2.cvtColor(patente_original, cv2.COLOR_RGB2GRAY)

_, thresh = cv2.threshold(patente_gray, 100, 255, cv2.THRESH_BINARY)

imshow(thresh, title="Original")


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(patente_gray)
# Filtrar componentes con área mayor a 300
filtered_labels = np.zeros_like(labels, dtype=np.uint8)  # Crear una nueva máscara
for i in range(1, num_labels):  # Ignorar el fondo (etiqueta 0)
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 300 or area > 150000:
        filtered_labels[labels == i] = 255  # Mantener el componente
    if area < 300 or area > 150000:
        filtered_labels[labels == i] = 255  # Mantener el componente
# imshow(filtered_labels, title='Original')

imshow(labels, title="Original")

# Convertir la imagen binaria a una RGB para visualizar
imagen = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

# Encontrar contornos y jerarquía
contours, hierarchy = cv2.findContours(
    filtered_labels, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

contornos_rectangulares = []

# Iterar sobre cada contorno
for i, contour in enumerate(contours):
    # Calcular el área del contorno
    area = cv2.contourArea(contour)

    # Filtrar contornos por área
    if 15 < area < 200:
        # Aproximar el contorno a un polígono
        epsilon = 0.02 * cv2.arcLength(contour, True)  # Parámetro de aproximación
        # approx = cv2.approxPolyDP(contour, epsilon, True)

        # Verificar si el contorno aproximado tiene 4 vértices (rectángulo) if len(approx) == 4:
        # contornos_rectangulares.append(contour)

        # Opcional: Dibujar el rectángulo delimitador en la imagen
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Rectángulo azul

# Dibujar los contornos rectangulares en la imagen
cv2.drawContours(
    imagen, contornos_rectangulares, -1, (0, 255, 0), 3
)  # Verde para contornos rectangulares

# Mostrar la imagen con los rectángulos detectados
imshow(imagen, title="Rectángulos detectados")


imshow(output, title="Original")


# Cargar la imagen
patente = cv2.imread(f"img02.png", cv2.IMREAD_COLOR)
patente_gray = cv2.cvtColor(patente, cv2.COLOR_BGR2GRAY)


# Encontrar contornos
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Inicializar coordenadas del bounding box completo
x_min, y_min, x_max, y_max = float("inf"), float("inf"), float("-inf"), float("-inf")

# Iterar sobre cada contorno para encontrar el bounding box que los contiene a todos
contours_found = False
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Filtrar objetos con tamaño adecuado para una patente
    if (
        100 < w < 400 and 30 < h < 150 and 2 <= w / h <= 6
    ):  # Ajustar los umbrales y la relación de aspecto según sea necesario
        # Actualizar coordenadas del bounding box completo
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
        contours_found = True
        cv2.rectangle(
            patente, (x, y), (x + w, y + h), (0, 0, 255), 2
        )  # Dibujar cada contorno en rojo para ver su ubicación

# Convertir las coordenadas a enteros solo si se encontraron contornos válidos
if contours_found:
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    # Dibujar el bounding box completo en la imagen original
    cv2.rectangle(patente, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Mostrar el resultado
    cv2.imshow("Patente Detectada", patente)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se encontraron contornos válidos.")


imshow(roi, title="Original")

# Cargar la imagen
patente = cv2.imread(f"img02.png", cv2.IMREAD_COLOR)
patente_gray = cv2.cvtColor(patente, cv2.COLOR_BGR2GRAY)

patente_gray[patente_gray < 130] = 0

_, thresh = cv2.threshold(patente_gray, 100, 255, cv2.THRESH_BINARY)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
# Filtrar componentes con área mayor a 300
filtered_labels = np.zeros_like(labels, dtype=np.uint8)  # Crear una nueva máscara
for i in range(1, num_labels):  # Ignorar el fondo (etiqueta 0)
    area = stats[i, cv2.CC_STAT_AREA]
    if area < 50:
        filtered_labels[labels == i] = 255  # Mantener el componente


imshow(filtered_labels, title="Original")

# Encontrar contornos
contours, _ = cv2.findContours(filtered_labels, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

"""# Crear una copia de la imagen para dibujar los resultados
output = cv2.cvtColor(filtered_labels, cv2.COLOR_GRAY2BGR)

# Iterar sobre cada contorno for contour in contours:
    # Calcular el bounding box (rectángulo que encierra el objeto) x, y, w, h = cv2.boundingRect(contour)

    # Aplicar el filtro: mantener objetos con relación de aspecto entre 0.5 y 2 if 40 < w < 100 and 15 < h < 100:
        # Dibujar el contorno que pasa el filtro cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)"""

imagen = cv2.cvtColor(filtered_labels, cv2.COLOR_GRAY2RGB)
contornos_rectangulares = []

# Iterar sobre cada contorno
for i, contour in enumerate(contours):
    # Calcular el área del contorno
    area = cv2.contourArea(contour)

    # Filtrar contornos por área
    if 15 < area < 400:
        # Aproximar el contorno a un polígono epsilon = 0.02 * cv2.arcLength(contour, True)  # Parámetro de aproximación
        # approx = cv2.approxPolyDP(contour, epsilon, True)

        # Verificar si el contorno aproximado tiene 4 vértices (rectángulo) if len(approx) == 4:
        contornos_rectangulares.append(contour)

        # Opcional: Dibujar el rectángulo delimitador en la imagen x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(imagen, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Rectángulo azul

# Dibujar los contornos rectangulares en la imagen
cv2.drawContours(
    imagen, contornos_rectangulares, -1, (0, 255, 0), 3
)  # Verde para contornos rectangulares

imshow(imagen, title="Original")

imshow(mediana, title="Original")
