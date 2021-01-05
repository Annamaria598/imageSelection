import imageio
import numpy as np
import scipy.ndimage as nd
import sklearn
from skimage.color import rgb2lab, rgba2rgb, rgb2hsv
from scipy.stats import entropy
from sklearn.metrics import matthews_corrcoef

def _entropy(values):
    probabilities = np.bincount(values.astype(np.int)) / float(len(values))
    return entropy(probabilities)

def local_entropy(img, kernel_radius=2):
    """
    Calcolo dell'entropia per l'intorno di ogni pixel in
    un intorno specificato dal kernel.

    img: ndarray dell'immagine.
    kernel_radius: int, dimensione del kernel
    """
    return nd.generic_filter(img, _entropy, size=kernel_radius)


def image_to_data(img, size=13):
    """
    Genera ed associa attributi ad ogni pixel dell'immagine
    utilizzando vari filtri.

    Restituisce un dizionario di matrici. Ogni matrice è una feature/livello.

    img: str, il path ad un'immagine.
    size: int, la dimensione della finestra da utilizzare per i filtri.
    """

    # mettiamo tutti i layer/features in un dizionario
    layers = {}

    # apriamo immagine come matrice in RGB
    image_array = imageio.imread(img)

    # ----- SPAZIO COLORE RGB -----
    if (img.split(".")[-1] in ["png", "PNG"]):
        image_array = rgba2rgb(image_array)

    # separiamo le componenti R, G, B
    R = image_array[..., 0]
    G = image_array[..., 1]
    B = image_array[..., 2]

    # normalizziamo i valori per restare
    # attorno all'intervallo -0.5, 0.5
    # Se abbiamo usato rgba2rgb non è necessario
    # dividere per 255
    #R = R / 255
    layers['R'] = R - 0.5
    #G = G / 255
    layers['G'] = G - 0.5
    #B = B / 255
    layers['B'] = B - 0.5

    # calcolo filtro mediano
    layers['R_mf'] = nd.median_filter(R, size=size)
    layers['G_mf'] = nd.median_filter(G, size=size)
    layers['B_mf'] = nd.median_filter(B, size=size)

    # edge detection
    # usiamo per il momento i parametri di default
    # e ci riserviamo in futuro la possibilità di
    # sperimentare con parametri diversi
    layers['R_edge'] = nd.sobel(R)
    layers['G_edge'] = nd.sobel(G)
    layers['B_edge'] = nd.sobel(B)

    # varianza
    mean_R = nd.uniform_filter(R, (size, size))
    mean_G = nd.uniform_filter(G, (size, size))
    mean_B = nd.uniform_filter(B, (size, size))
    mean_sqr_R = nd.uniform_filter(R**2, (size, size))
    mean_sqr_G = nd.uniform_filter(G**2, (size, size))
    mean_sqr_B = nd.uniform_filter(B**2, (size, size))

    layers['R_variance'] = mean_sqr_R - mean_R**2
    layers['G_variance'] = mean_sqr_G - mean_G**2
    layers['B_variance'] = mean_sqr_B - mean_B**2

    # ----- SPAZIO COLORE LAB -----

    image_array_lab = rgb2lab(image_array)

    # separiamo le componenti L, a , b
    L = image_array_lab[..., 0]
    a = image_array_lab[..., 1]
    b = image_array_lab[..., 2]

    # Sfruttiamo la componente L per calcolare
    # un filtro entropia
    layers["entropy"] = local_entropy(L, kernel_radius=size)

    # Normalizzo l'entropia dividendo per il massimo valore
    # di entropia possibile su un segnale lungo size*size
    # con 100 simboli a disposizione
    # questo calcolo del massimo per cui dividere è valido solo
    # per una finestra con un massimo di 200 pixel
    signal_size = size**2
    if signal_size > 100:
        max_ent = entropy(list(range(100)) + [i for i in range(signal_size - 100)])
    else:
        max_ent = entropy(list(range(signal_size)))
    layers["entropy"] = ( layers["entropy"] / max_ent ) - 0.5

    # normalizziamo i valori per restare
    # attorno all'intervallo -0.5, 0.5
    # L va da 0 a 100
    # a e b vanno da -120 a +120
    L = L / 100
    layers['L'] = L - 0.5
    layers['a'] = a / 240
    layers['b'] = b / 240

    # calcolo filtro mediano
    layers['L_mf'] = nd.median_filter(L, size=size)
    layers['a_mf'] = nd.median_filter(a, size=size)
    layers['b_mf'] = nd.median_filter(b, size=size)

    # edge detection
    # usiamo per il momento i parametri di default
    # e ci riserviamo in futuro la possibilità di
    # sperimentare con parametri diversi
    layers['L_edge'] = nd.sobel(L)
    layers['a_edge'] = nd.sobel(a)
    layers['b_edge'] = nd.sobel(b)

    # varianza
    mean_L = nd.uniform_filter(L, (size, size))
    mean_a = nd.uniform_filter(a, (size, size))
    mean_b = nd.uniform_filter(b, (size, size))
    mean_sqr_L = nd.uniform_filter(L**2, (size, size))
    mean_sqr_a = nd.uniform_filter(a**2, (size, size))
    mean_sqr_b = nd.uniform_filter(b**2, (size, size))

    layers['L_variance'] = mean_sqr_L - mean_L**2
    layers['a_variance'] = mean_sqr_a - mean_a**2
    layers['b_variance'] = mean_sqr_b - mean_b**2

    # ----- SPAZIO COLORE HSV -----

    image_array_hsv = rgb2hsv(image_array)

    # separiamo le componenti H, S, V
    h = image_array_hsv[..., 0]
    s = image_array_hsv[..., 1]
    v = image_array_hsv[..., 2]

    # normalizziamo i valori per restare
    # attorno all'intervallo -0.5, 0.5
    h = h / 360
    layers['h'] = h - 0.5
    layers['s'] = s - 0.5
    layers['v'] = v - 0.5

    # calcolo filtro mediano
    layers['h_mf'] = nd.median_filter(h, size=size)
    layers['s_mf'] = nd.median_filter(s, size=size)
    layers['v_mf'] = nd.median_filter(v, size=size)

    # edge detection
    # usiamo per il momento i parametri di default
    # e ci riserviamo in futuro la possibilità di
    # sperimentare con parametri diversi
    layers['h_edge'] = nd.sobel(h)
    layers['s_edge'] = nd.sobel(s)
    layers['v_edge'] = nd.sobel(v)

    # varianza
    mean_h = nd.uniform_filter(h, (size, size))
    mean_s = nd.uniform_filter(s, (size, size))
    mean_v = nd.uniform_filter(v, (size, size))
    mean_sqr_h = nd.uniform_filter(h**2, (size, size))
    mean_sqr_s = nd.uniform_filter(s**2, (size, size))
    mean_sqr_v = nd.uniform_filter(v**2, (size, size))

    layers['h_variance'] = mean_sqr_h - mean_h**2
    layers['s_variance'] = mean_sqr_s - mean_s**2
    layers['v_variance'] = mean_sqr_v - mean_v**2

    return layers


def pixel_to_class(pixel):
    """
    Assegna ad un singolo pixel una label intera in base al
    colore di cui è stato colorato.
    """

    if (pixel == [255, 255, 255, 255]).all(): # pixel del vetrino
        return 0
    elif (pixel == [163, 167, 69, 255]).all(): # pixel dello strato corneo
        return 1
    elif (pixel == [0, 255, 255, 255]).all(): # pixel del derma
        return 2
    elif (pixel == [25, 55, 190, 255]).all(): # pixel dell'epidermide
        return 3
    else: # pixel dei vasi
        return 4


def colimage_to_classes(img):
    """
    Crea le classi per ogni pixel a partire da un'immagine colorata.

    img: str, uri dell'immagine colorata.
    """

    img_col = imageio.imread(img)

    classes_matrix = []

    for l in range(len(img_col)):
        for c in range(len(img_col[0])):
            classes_matrix.append(pixel_to_class(img_col[l][c]))

    classes_matrix = np.array(classes_matrix)

    return classes_matrix


def class_to_pixel(intClass):
    """
    Converte una classe rappresentata da un intero
    in una lista di componenti colore.
    
    intClass: int, classe utilizzata come label nei
        dati di test o training.
    """
    
    if intClass == 0: # vetrino
        return [255, 255, 255, 255]
    elif intClass == 1: # strato corneo
        return [163, 167, 69, 255]
    elif intClass == 2: # epidermide
        return [0, 255, 255, 255]
    elif intClass == 3: # derma
        return [25, 55, 190, 255]
    else: # vasi
        return [255, 0, 255, 255]


def classes_to_colimage(y, shape):
    """
    Converte un'array di classi y in un'immagine
    da visualizzare con i colori originali.

    y: numpy array, array unidimensionale di classi
        associate ai pixel di un'immagine.
    shape: iterable, la forma dell'immagine originale.
    """

    pixel_list = []

    for pixel in y:
        pixel_list.append(class_to_pixel(pixel))
        
    pixel_list = np.array(pixel_list)
    
    final_image = pixel_list.reshape(shape)

    return final_image