import matplotlib

# Configuramos el uso de TkAgg como backend para matplotlib
matplotlib.use("TkAgg")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# Definimos una función personalizada para mostrar imágenes
def imshow(
    img,
    new_fig=True,
    title=None,
    color_img=False,
    blocking=False,
    colorbar=False,
    ticks=False,
):
    # Si new_fig es True, crea una nueva figura
    if new_fig:
        plt.figure()
    # Si color_img es True, muestra la imagen en su color original
    if color_img:
        plt.imshow(img)
    else:
        # Si no, muestra la imagen en escala de grises
        plt.imshow(img, cmap="gray")
    # Agregar título a la imagen
    plt.title(title)
    # Si no se desea mostrar los ticks, se eliminan
    if not ticks:
        plt.xticks([]), plt.yticks([])
    # Si se pide, añadir una barra de colores
    if colorbar:
        plt.colorbar()
    # Mostrar la imagen con la opción de bloqueo si es necesario
    if new_fig:
        plt.show(block=blocking)


# Leer la imagen de monedas
moneda = cv2.imread("monedas.jpg", cv2.IMREAD_COLOR)
# Convertir la imagen de BGR a RGB
moneda_original = cv2.cvtColor(moneda, cv2.COLOR_BGR2RGB)
# Convertir la imagen a escala de grises
img_fil_gray = cv2.cvtColor(moneda_original, cv2.COLOR_RGB2GRAY)
# Aplicar un filtro gaussiano para suavizar la imagen
filtrado = cv2.GaussianBlur(img_fil_gray, (5, 5), 0)
# Detectar bordes utilizando el algoritmo de Canny
canny = cv2.Canny(filtrado, 75, 150)
# Aplicar dilatación a la imagen de bordes para agrandar los objetos
dilatacion = cv2.dilate(canny, np.ones((13, 13), np.uint8))
# Realizar una operación de cierre morfológico para rellenar huecos
img_modif = cv2.morphologyEx(dilatacion, cv2.MORPH_CLOSE, np.ones((27, 27), np.uint8))


#imshow(dilatacion, title="Original")  # Descomentar si deseas ver la dilatación


# Función para realizar la reconstrucción morfológica de una imagen
def imreconstruct(marker, mask, kernel=None):
    # Asegurarse de que 'marker' y 'mask' tengan el mismo tamaño y tipo
    if marker.shape != mask.shape:
        raise ValueError("El tamaño de 'marker' y 'mask' debe ser igual")
    if marker.dtype != mask.dtype:
        marker = marker.astype(mask.dtype)

    # Si no se proporciona un kernel, utilizar uno por defecto (3x3)
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)

    while True:
        # Dilatación del marcador
        expanded = cv2.dilate(marker, kernel)

        # Realizar una intersección entre la imagen dilatada y la máscara
        expanded_intersection = cv2.bitwise_and(expanded, mask)

        # Verificar si la reconstrucción ha convergido (si no hay cambios)
        if np.array_equal(marker, expanded_intersection):
            break

        # Actualizar el marcador para la siguiente iteración
        marker = expanded_intersection

    return expanded_intersection


# Función para rellenar huecos en una imagen binaria
def imfillhole(img):
    # Crear una máscara de ceros para los bordes de la imagen
    mask = np.zeros_like(img)
    # Generar bordes alrededor de la máscara para realizar la operación de dilatación
    mask = cv2.copyMakeBorder(
        mask[1:-1, 1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)
    )
    # El marcador es el complemento de los bordes
    marker = cv2.bitwise_not(img, mask=mask)
    # La máscara es el complemento de la imagen original
    img_c = cv2.bitwise_not(img)
    # Realizar la reconstrucción morfológica
    img_r = imreconstruct(marker=marker, mask=img_c)
    # La imagen con los huecos rellenos es el complemento de la reconstrucción
    img_fh = cv2.bitwise_not(img_r)
    return img_fh


# Función para encontrar el menor factor de forma
def menor_factor_de_forma(factor):
    primero = 1  # Valor inicial más grande para el factor de forma
    indiceuno = 0
    for registro in factor:
        if registro[1] < primero:  # Si encontramos un valor más pequeño
            primero = registro[1]
            indiceuno = registro[0]
    factor.pop(indiceuno - 1)  # Eliminar el elemento con el factor más pequeño
    return indiceuno, primero, factor


# Función para encontrar el área más pequeña en una lista de monedas
def menor_area(monedas):
    area_chica = 100000000  # Valor inicial muy grande para el área
    for i in range(len(monedas)):
        if monedas[i][1] < area_chica:  # Si encontramos un área más pequeña
            area_chica = monedas[i][1]
            indice = i
    return indice


# Llenar los huecos en la imagen de las monedas
relleno = imfillhole(img_modif)

#imshow(relleno, title="Moneda Rellenada")  # Descomentar si deseas ver la imagen rellena

# Aplicar una operación de erosión a la imagen de las monedas rellenas
erocion = cv2.erode(relleno, np.ones((41, 41), np.uint8))

#imshow(erocion, title="Monedas Erocionadas")  # Descomentar si deseas ver la erosión

# Inicializar las listas para el factor de forma y las cajas
factordeforma = []
caja = []
# Obtener los componentes conectados en la imagen de las monedas erosionadas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erocion)
for i_label in range(1, num_labels):
    area = stats[i_label, cv2.CC_STAT_AREA]
    filtered_labels = np.zeros_like(erocion, dtype=np.uint8)
    filtered_labels[labels == i_label] = 255
    # Encontrar los contornos de los componentes
    contours, _ = cv2.findContours(
        filtered_labels, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    for contour in contours:
        # Calcular el bounding box (caja delimitadora)
        x, y, w, h = cv2.boundingRect(contour)
        perimetro = cv2.arcLength(contour, True)  # Perímetro del contorno
        factor_forma = area / (perimetro**2)  # Calcular el factor de forma
        factordeforma.append([i_label, factor_forma])  # Guardar factor de forma
        caja.append([i_label, area, x, y, w, h])  # Guardar información de la caja

# Encontrar el menor factor de forma (más pequeño)
indice, factor, factordeforma2 = menor_factor_de_forma(factordeforma)
indice2, factor2, factordeforma3 = menor_factor_de_forma(factordeforma2)

# Seleccionar las monedas con los menores factores de forma
monedas = []
for i in range(len(factordeforma3)):
    for j in range(len(caja)):
        if factordeforma3[i][0] == caja[j][0]:
            monedas.append(caja[j])
            continue

# Seleccionar las cajas de dados
dados = []
for i in range(len(caja)):
    if caja[i][0] == indice:
        dados.append(caja[i])
    elif caja[i][0] == indice2:
        dados.append(caja[i])

# Ahora solo las monedas quedan, y vamos a seleccionar las más pequeñas (monedas de 10 centavos)
monedas_etiqueta = []
for i in range(len(monedas)):
    indice = menor_area(monedas)
    # Etiquetar las monedas
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

# Realizar dilatación y seleccionar los dados
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

# Erosionar los dados para mejorar la detección
erocion_dados = cv2.erode(delatacion_copia, np.ones((11, 11), np.uint8))
relleno_dado = imfillhole(erocion_dados)

# Erosionar nuevamente
erocion_dados2 = cv2.erode(relleno_dado, np.ones((11, 11), np.uint8))

# Obtener los componentes conectados en la imagen de los dados
num_labels_dado, labels_dado, stats_dado, centroids_dado = (
    cv2.connectedComponentsWithStats(erocion_dados2)
)
foto = np.zeros_like(relleno_dado, dtype=np.uint8)
for i_label in range(1, num_labels_dado):
    area_dado = stats_dado[i_label, cv2.CC_STAT_AREA]
    if area_dado > 1000:
        foto[labels_dado == i_label] = 255

# Etiquetar las caras de los dados
dado_etiqueta = []
cara_dado1 = cv2.connectedComponentsWithStats(
    foto[  # Imagen recortada para el primer dado
        dados[0][3] : dados[0][3] + dados[0][5], dados[0][2] : dados[0][2] + dados[0][4]
    ]
)
cara_dado2 = cv2.connectedComponentsWithStats(
    foto[  # Imagen recortada para el segundo dado
        dados[1][3] : dados[1][3] + dados[1][5], dados[1][2] : dados[1][2] + dados[1][4]
    ]
)

# Guardar las etiquetas para los dados
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

imshow(copia, title="Dados y monedas etiquetados por valor", blocking=True)

