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


# Lista de nombres de imágenes

# for i in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]: Leer la imagen
for i in ["01", "02"]:
    patente = cv2.imread(f"img{i}.png", cv2.IMREAD_COLOR)
    patente_original = cv2.cvtColor(patente, cv2.COLOR_BGR2RGB)
    patente_gray = cv2.cvtColor(patente_original, cv2.COLOR_RGB2GRAY)

    # Umbral binario
    _, thresh = cv2.threshold(patente_gray, 100, 255, cv2.THRESH_BINARY)
    imshow(thresh, title="imagen binaria")
    # ---------------------------------------------------------------------------------------------------------------------------

    # Componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    # Filtrar componentes con área
    filtered_labels = np.zeros_like(labels, dtype=np.uint8)
    for i_label in range(1, num_labels):  # Ignorar el fondo (etiqueta 0)
        area = stats[i_label, cv2.CC_STAT_AREA]
        if area < 300 or 1000 < area < 2000 or area > 75000:
            filtered_labels[labels == i_label] = 255
    imshow(filtered_labels, title="imagen filtrada por el area de los componentes")
    # ---------------------------------------------------------------------------------------------------------------------------

    # Encontrar contornos
    contours, _ = cv2.findContours(
        filtered_labels, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Crear una nueva imagen binaria para guardar las regiones seleccionadas
    binary_output = np.zeros_like(patente_gray, dtype=np.uint8)

    # Iterar sobre cada contorno
    for contour in contours:
        # Calcular el bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Aplicar el filtro: mantener objetos con relación de aspecto entre 0.5 y 2
        if 4 < w < 150 and 4 < h < 150:
            # Dibujar el rectángulo en la imagen binaria
            binary_output[y : y + h, x : x + w] = thresh[y : y + h, x : x + w]

    # Mostrar la imagen binaria resultante
    imshow(binary_output, title="Imagen con Regiones Binarias Filtradas")

    # ---------------------------------------------------------------------------------------------------------------------------

    # Componentes conectados
    num_labels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(
        binary_output
    )

    # Filtrar componentes con área
    filtered_labels2 = np.zeros_like(labels2, dtype=np.uint8)
    for i_label2 in range(1, num_labels2):
        area2 = stats2[i_label2, cv2.CC_STAT_AREA]
        if 4 < area2 < 200 or 600 < area2 < 700 or 1000 < area2 < 2300:
            filtered_labels2[labels2 == i_label2] = 255

    imshow(filtered_labels2, title="Imagen filtrada por el area de los componentes 2")

    # ---------------------------------------------------------------------------------------------------------------------------

    # Encontrar contornos
    contours2, _ = cv2.findContours(
        filtered_labels2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Crear una nueva imagen binaria para guardar las regiones seleccionadas
    binary_output2 = np.zeros_like(patente_gray, dtype=np.uint8)

    # Crear una copia de la imagen para dibujar los resultados
    output2 = cv2.cvtColor(filtered_labels2, cv2.COLOR_GRAY2RGB)

    bounding_boxes = []
    # Iterar sobre cada contorno
    for contour2 in contours2:
        # Calcular el bounding box
        x2, y2, w2, h2 = cv2.boundingRect(contour2)

        if i in ["01", "04", "05", "06", "08", "11"]:
            if 64 < w2 < 100 and 24 < h2 < 40:
                # Dibujar el rectángulo en la imagen binaria
                bounding_boxes.append([x2, y2, w2, h2])
                binary_output2[y2 : y2 + h2, x2 : x2 + w2] = thresh[
                    y2 : y2 + h2, x2 : x2 + w2
                ]
        else:
            if 4 < w2 < 20 and 8 < h2 < 20:
                bounding_boxes.append([x2, y2, w2, h2])

    if i in ["01", "04", "05", "06", "08", "11"]:
        foto_cajas = cv2.cvtColor(binary_output2, cv2.COLOR_GRAY2RGB)
        for x8, y8, w8, h8 in bounding_boxes:
            cv2.rectangle(foto_cajas, (x8, y8), (x8 + w8, y8 + h8), (0, 255, 0), 2)
    else:
        # Filtrar por cercanía
        filtered_boxes = []

        for h, (x4, y4, w4, h4) in enumerate(bounding_boxes):
            cx2 = x4 + (w4 / 2)
            cy2 = y4 + (h4 / 2)
            filtered_boxes.append([h, cx2, cy2])
        cajas_finales = []

        for j in range(len(filtered_boxes)):
            cajas = []
            cajas.append(bounding_boxes[j])
            for k in range(j + 1, len(filtered_boxes)):
                resta_x = abs(filtered_boxes[j][1] - filtered_boxes[k][1])
                resta_y = abs(filtered_boxes[j][2] - filtered_boxes[k][2])
                if 0.0 < resta_x < 80.0 and 0.0 < resta_y < 20.0:
                    cajas.append(bounding_boxes[k])

            if len(cajas) > 4:
                cajas_finales = cajas_finales + cajas
                break

        # Dibujar las cajas filtradas
        for x8, y8, w8, h8 in cajas_finales:
            binary_output2[y8 : y8 + h8, x8 : x8 + w8] = thresh[
                y8 : y8 + h8, x8 : x8 + w8
            ]

        foto_cajas = cv2.cvtColor(binary_output2, cv2.COLOR_GRAY2RGB)
        for x8, y8, w8, h8 in cajas_finales:
            cv2.rectangle(foto_cajas, (x8, y8), (x8 + w8, y8 + h8), (0, 255, 0), 2)
    imshow(foto_cajas, title="Patentes del vehiculo")
