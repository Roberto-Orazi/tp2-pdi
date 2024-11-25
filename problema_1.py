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
kernel = np.ones((3, 3), np.uint8)
gradiente = cv2.morphologyEx(img_fil_gray, cv2.MORPH_GRADIENT, kernel)

_, thresh = cv2.threshold(gradiente, 150, 255, cv2.THRESH_OTSU)
# imshow(moneda_original, title='Original')

imshow(thresh, title="Original")


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


# hacemos un filtrado para elimimar ruido con sal
def charmean(imgn, k=3, Q=1.5):
    imgn_f = imgn.astype(np.float64)
    if Q < 0:
        imgn_f += np.finfo(float).eps
    w = np.ones((k, k))
    I_num = cv2.filter2D(imgn_f ** (Q + 1), cv2.CV_64F, w)
    I_den = cv2.filter2D(imgn_f**Q, cv2.CV_64F, w)
    I = I_num / (I_den + np.finfo(float).eps)
    return I


imgn_ch_filt = charmean(thresh, 2, -1.5)
imgn_ch_filt8 = imgn_ch_filt.astype(np.uint8)


imshow(imgn_ch_filt8, title="Original")

img_fh = imfillhole(imgn_ch_filt8)

imshow(img_fh, title="Original")


img_modif = cv2.morphologyEx(img_fh, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
dilatacion = cv2.dilate(img_modif, np.ones((13, 13), np.uint8))


imshow(dilatacion, title="Original")


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilatacion)
# Filtrar componentes con área mayor a 300
filtered_labels = np.zeros_like(labels, dtype=np.uint8)  # Crear una nueva máscara
for i in range(1, num_labels):  # Ignorar el fondo (etiqueta 0)
    area = stats[i, cv2.CC_STAT_AREA]
    if area > 950:
        filtered_labels[labels == i] = 255  # Mantener el componente


imshow(filtered_labels, title="Original")


img_modif3 = cv2.morphologyEx(
    filtered_labels, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
)
imshow(img_modif3, title="Original")

img_fh2 = imfillhole(img_modif3)
imshow(img_fh2, title="Original")


erocion = cv2.erode(img_fh2, np.ones((41, 41), np.uint8))
imshow(erocion, title="Original")

num_labels2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(erocion)

cent10 = 0
peso1 = 0
cent50 = 0

# Filtrar componentes con área mayor a 300
centavos10 = np.zeros_like(labels2, dtype=np.uint8)  # Crear una nueva máscara
for i in range(1, num_labels2):  # Ignorar el fondo (etiqueta 0)
    area = stats2[i, cv2.CC_STAT_AREA]
    if 40000 < area < 47000:
        cent10 += 1
        centavos10[labels2 == i] = 255  # Mantener el componente

imshow(centavos10, title="Original")

# Filtrar componentes con área mayor a 300
peso = np.zeros_like(labels2, dtype=np.uint8)  # Crear una nueva máscara
for i in range(1, num_labels2):  # Ignorar el fondo (etiqueta 0)
    area = stats2[i, cv2.CC_STAT_AREA]
    if 60000 < area < 70000:
        peso1 += 1
        peso[labels2 == i] = 255  # Mantener el componente

imshow(peso, title="Original")


# Filtrar componentes con área mayor a 300
centavos50 = np.zeros_like(labels2, dtype=np.uint8)  # Crear una nueva máscara
for i in range(1, num_labels2):  # Ignorar el fondo (etiqueta 0)
    area = stats2[i, cv2.CC_STAT_AREA]
    if 78000 < area < 83000:
        cent50 += 1
        centavos50[labels2 == i] = 255  # Mantener el componente

imshow(centavos50, title="Original")
print(f"Tenemos un total de {cent10} monedas de 10 centavos")
print(f"Tenemos un total de {cent50} monedas de 50 centavos")
print(f"Tenemos un total de {peso1} monedas de un peso")
print(f"Tenemos un TOTAL de: {cent10*0.10+cent50*0.50+peso1} ")


num_labels3, labels3, stats3, centroids3 = cv2.connectedComponentsWithStats(
    imgn_ch_filt8
)

# Filtrar componentes con área mayor a 300
filtered_labels3 = np.zeros_like(labels3, dtype=np.uint8)  # Crear una nueva máscara
for i in range(1, num_labels3):  # Ignorar el fondo (etiqueta 0)
    area = stats3[i, cv2.CC_STAT_AREA]
    if area < 800:
        filtered_labels3[labels3 == i] = 255  # Mantener el componente


imshow(filtered_labels3, title="Original")

img_fh_dado = imfillhole(filtered_labels3)

erocion_dado = cv2.erode(img_fh_dado, np.ones((17, 17), np.uint8))
imshow(erocion_dado, title="Original")

num_labels4, labels4, stats4, centroids4 = cv2.connectedComponentsWithStats(
    erocion_dado
)
print(f"Tenemos 2 dados que la suma de sus caras suman {num_labels4-1}")
