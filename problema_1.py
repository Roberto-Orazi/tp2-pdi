import matplotlib

matplotlib.use("TkAgg")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


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


moneda = cv2.imread("monedas.jpg", cv2.IMREAD_COLOR)
moneda_original = cv2.cvtColor(moneda, cv2.COLOR_BGR2RGB)
img_fil_gray = cv2.cvtColor(moneda_original, cv2.COLOR_RGB2GRAY)
filtrado = cv2.GaussianBlur(img_fil_gray, (5, 5), 0)
canny = cv2.Canny(filtrado, 75, 150)
dilatacion = cv2.dilate(canny, np.ones((13, 13), np.uint8))
img_modif = cv2.morphologyEx(dilatacion, cv2.MORPH_CLOSE, np.ones((27, 27), np.uint8))


imshow(dilatacion, title="Original")


def imreconstruct(marker, mask, kernel=None):
    # Asegurarse de que marker y mask sean del mismo tamaño y tipo
    if marker.shape != mask.shape:
        raise ValueError("El tamaño de 'marker' y 'mask' debe ser igual")
    if marker.dtype != mask.dtype:
        marker = marker.astype(mask.dtype)

    # Definir el kernel si no se proporciona
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)

    while True:
        # Dilatación
        expanded = cv2.dilate(marker, kernel)

        # Intersección entre la imagen dilatada y la máscara
        expanded_intersection = cv2.bitwise_and(expanded, mask)

        # Verificar si la reconstrucción ha convergido
        if np.array_equal(marker, expanded_intersection):
            break

        # Actualizar el marcador
        marker = expanded_intersection

    return expanded_intersection


def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)  # Genero mascara para...
    mask = cv2.copyMakeBorder(
        mask[1:-1, 1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)
    )  # ... seleccionar los bordes.
    marker = cv2.bitwise_not(
        img, mask=mask
    )  # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(
        img
    )  # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(
        marker=marker, mask=img_c
    )  # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(
        img_r
    )  # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh


def menor_factor_de_forma(factor):
    primero = 1
    indiceuno = 0
    for registro in factor:
        if registro[1] < primero:
            primero = registro[1]
            indiceuno = registro[0]
    factor.pop(indiceuno - 1)
    return indiceuno, primero, factor


def menor_area(monedas):
    area_chica = 100000000
    for i in range(len(monedas)):
        if monedas[i][1] < area_chica:
            indice = i
    return indice


relleno = imfillhole(img_modif)

imshow(relleno, title="Original")

erocion = cv2.erode(relleno, np.ones((41, 41), np.uint8))

imshow(erocion, title="Original")

factordeforma = []
caja = []
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erocion)
for i_label in range(1, num_labels):
    area = stats[i_label, cv2.CC_STAT_AREA]
    filtered_labels = np.zeros_like(erocion, dtype=np.uint8)
    filtered_labels[labels == i_label] = 255
    contours, _ = cv2.findContours(
        filtered_labels, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    for contour in contours:
        # Calcular el bounding box
        x, y, w, h = cv2.boundingRect(contour)
        perimetro = cv2.arcLength(contour, True)  # True porque el contorno está cerrado
        factor_forma = area / (perimetro**2)
        factordeforma.append([i_label, factor_forma])
        caja.append([i_label, area, x, y, w, h])

# Los factores de forma mas chicos son los dados
indice, factor, factordeforma2 = menor_factor_de_forma(factordeforma)
indice2, factor2, factordeforma3 = menor_factor_de_forma(factordeforma2)


monedas = []
for i in range(len(factordeforma3)):
    for j in range(len(caja)):
        if factordeforma3[i][0] == caja[j][0]:
            monedas.append(caja[j])
            continue

dados = []
for i in range(len(caja)):
    if caja[i][0] == indice:
        dados.append(caja[i])
    elif caja[i][0] == indice2:
        dados.append(caja[i])

# Ahora solo los quedo las monedas y vamos a seleccionar las areas mas chicas , las cuales son las monedas de 10
# centavos
monedas_etiqueta = []
# Sabemos que son 9 monedas, las cuales son las mas chicas son de 10 centavos
for i in range(len(monedas)):
    indice = menor_area(monedas)
    if 0 <= i <= 8:
        registro = monedas[indice]
        monedas_etiqueta.append(
            ["10 centavos", registro[2], registro[3], registro[4], registro[5]]
        )
    if 9 <= i <= 13:
        registro = monedas[indice]
        monedas_etiqueta.append(
            ["1 peso", registro[2], registro[3], registro[4], registro[5]]
        )
    if 14 <= i <= 16:
        registro = monedas[indice]
        monedas_etiqueta.append(
            ["50 centavos", registro[2], registro[3], registro[4], registro[5]]
        )
    monedas.pop(indice)

delatacion_copia = np.zeros_like(dilatacion, dtype=np.uint8)
delatacion_copia[
    dados[0][3] : dados[0][3] + dados[0][5], dados[0][2] : dados[0][2] + dados[0][4]
] = dilatacion[
    dados[0][3] : dados[0][3] + dados[0][5], dados[0][2] : dados[0][2] + dados[0][4]
]
delatacion_copia[
    dados[1][3] - 20 : dados[1][3] + dados[1][5],
    dados[1][2] : dados[1][2] + dados[1][4],
] = dilatacion[
    dados[1][3] - 20 : dados[1][3] + dados[1][5],
    dados[1][2] : dados[1][2] + dados[1][4],
]

erocion_dados = cv2.erode(delatacion_copia, np.ones((11, 11), np.uint8))
relleno_dado = imfillhole(erocion_dados)

erocion_dados2 = cv2.erode(relleno_dado, np.ones((11, 11), np.uint8))
imshow(erocion_dados2, title="Original")

num_labels_dado, labels_dado, stats_dado, centroids_dado = (
    cv2.connectedComponentsWithStats(erocion_dados2)
)
foto = np.zeros_like(relleno_dado, dtype=np.uint8)
for i_label in range(1, num_labels_dado):
    area_dado = stats_dado[i_label, cv2.CC_STAT_AREA]
    if area_dado > 1000:
        foto[labels_dado == i_label] = 255
dado_etiqueta = []
cara_dado1 = cv2.connectedComponentsWithStats(
    foto[
        dados[0][3] : dados[0][3] + dados[0][5], dados[0][2] : dados[0][2] + dados[0][4]
    ]
)
cara_dado2 = cv2.connectedComponentsWithStats(
    foto[
        dados[1][3] : dados[1][3] + dados[1][5], dados[1][2] : dados[1][2] + dados[1][4]
    ]
)

dado_etiqueta.append(
    [
        f"Valor de la cara = {cara_dado1[0]-1}",
        dados[0][2],
        dados[0][3],
        dados[0][4],
        dados[0][5],
    ]
)
dado_etiqueta.append(
    [
        f"Valor de la cara = {cara_dado2[0]-1}",
        dados[1][2],
        dados[1][3],
        dados[1][4],
        dados[1][5],
    ]
)

copia = moneda_original.copy()
for etiqueta, x, y, ancho, alto in monedas_etiqueta:
    # Coordenadas del rectángulo
    punto1 = (x, y)  # Esquina superior izquierda
    punto2 = (x + ancho, y + alto)  # Esquina inferior derecha

    # Dibujar el rectángulo
    cv2.rectangle(copia, punto1, punto2, color=(0, 255, 0), thickness=10)

    # Añadir la etiqueta (texto)
    cv2.putText(
        copia,
        etiqueta,
        (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        color=(0, 255, 0),
        thickness=10,
    )

for etiqueta, x, y, ancho, alto in dado_etiqueta:
    # Coordenadas del rectángulo
    punto1 = (x, y)  # Esquina superior izquierda
    punto2 = (x + ancho, y + alto)  # Esquina inferior derecha

    # Dibujar el rectángulo
    cv2.rectangle(copia, punto1, punto2, color=(0, 255, 0), thickness=10)

    # Añadir la etiqueta (texto)
    cv2.putText(
        copia,
        etiqueta,
        (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        color=(0, 255, 0),
        thickness=10,
    )

imshow(copia, title="Original")
