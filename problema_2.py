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
for num_foto in [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
]:
    # Leer la imagen
    patente = cv2.imread(f"img{num_foto}.png", cv2.IMREAD_COLOR)
    patente_original = cv2.cvtColor(patente, cv2.COLOR_BGR2RGB)
    patente_gray = cv2.cvtColor(patente_original, cv2.COLOR_RGB2GRAY)
    # umbral para hacer iterativo el metodo de deteccion de patentes y ademas personalizado para cada patente
    umbral = 50
    bandera = True
    while bandera:
        # Umbral binario
        if umbral > 250:
            bandera = False
            break

        _, thresh = cv2.threshold(patente_gray, umbral, 255, cv2.THRESH_BINARY)

        # imshow(thresh, title='imagen binaria')

        # ---------------------------------------------------------------------------------------------------------------------------

        # Componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, 4
        )
        area_menor = []
        l_aspecto = []
        # Filtrar componentes con área
        filtered_labels = np.zeros_like(labels, dtype=np.uint8)
        for i_label in range(1, num_labels):  # Ignorar el fondo (etiqueta 0)
            area = stats[i_label, cv2.CC_STAT_AREA]
            if 17 < area < 200:
                area_menor.append(i_label)
                relacion_aspecto = stats[i_label][3] / stats[i_label][2]
                if 1.5 <= relacion_aspecto <= 3.0:
                    l_aspecto.append(i_label)
        # poruqe tenemos que son 6 las componentes de la patente
        if len(l_aspecto) < 6:
            umbral = umbral + 1
            continue
        else:
            centro_caja = []
            for j in l_aspecto:
                x, y, w, h, a = stats[j]
                cx2 = x + (w / 2)
                cy2 = y + (h / 2)
                centro_caja.append([j, cx2, cy2])

            for k in range(len(centro_caja)):
                j, cx, cy = centro_caja[k]
                cajas_finales = []
                cajas_finales.append(j)
                for f in range(len(centro_caja)):
                    if f != k:
                        j3, cx3, cy3 = centro_caja[f]
                        resta_x = abs(cx - cx3)
                        resta_y = abs(cy - cy3)
                    if 0.0 < resta_x < 80.0 and 0.0 < resta_y < 20.0:
                        cajas_finales.append(j3)
                if len(cajas_finales) > 6 and len(cajas_finales) < 8:
                    bandera = False
                    break
        if bandera:
            umbral = umbral + 1

    # foto = np.zeros_like(thresh, dtype=np.uint8)
    esq_x_menor = 1000
    esq_y_menor = 1000
    esq_x_mayor = 0
    esq_y_mayor = 0
    copia_foto_original = patente_original.copy()
    for num in cajas_finales:
        x, y, w, h, a = stats[num]
        if a > 10:
            # Coordenadas del rectángulo
            punto1 = (x, y)  # Esquina superior izquierda
            punto2 = (x + w, y + h)  # Esquina inferior derecha
            # Dibujar el rectángulo
            cv2.rectangle(
                copia_foto_original, punto1, punto2, color=(0, 255, 0), thickness=1
            )
            if x < esq_x_menor:
                esq_x_menor = x
            if y < esq_y_menor:
                esq_y_menor = y

            if x + w > esq_x_mayor:
                esq_x_mayor = x + w
            if y + h > esq_y_mayor:
                esq_y_mayor = y + h
    punto1 = (esq_x_menor - 5, esq_y_menor - 5)  # Esquina superior izquierda
    punto2 = (esq_x_mayor + 5, esq_y_mayor + 5)  # Esquina inferior derecha
    cv2.rectangle(copia_foto_original, punto1, punto2, color=(255, 0, 0), thickness=2)

    imshow(copia_foto_original, title="imagen filtrada por el area de los componentes")
# ---------------------------------------------------------------------------------------------------------------------------
