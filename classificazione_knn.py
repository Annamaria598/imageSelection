import imageio 
import numpy as np
import scipy.ndimage as nd 
import sklearn
from skimage.color import rgb2lab, rgba2rgb, rgb2hsv
from scipy.stats  import entropy

def _entropy(values):
    probabilities = np.bincount(values.astype(np.int)) / float(len(values))
    return entropy(probabilities)

def local_entropy(img, kernel_radius=2):

    """
    Calcolo dell'entropia per l'intorno di ogni pixel in un intorno specificato del kernel.

    img: ndarray dell'immagine.
    kernel_radius : int, dimensione del kernel 
    """

    return nd.generic_filter(img, _entropy, size = kernel_radius)

def image_to_data(img, size = 13):
    """
    Genera e associa attributi ad ogni pixel dell'immagine utilizzando vari filtri.

    img: str, il path ad un'immagine.
    size : int, la dimensione della finestra da utilizzare per i filtri
    """

    # apriamo immagine come matrice in RGB
    image_array = imageio.imread(img)

# ----- SPAZIO COLORE RGB ----

    if(img.split(".")[-1] in ["png", "PNG"]):
        image_array = rgba2rgb(image_array)

    
    #separiamo le componenti R,G,B
    R = image_array[..., 0]
    G = image_array[..., 1]
    B = image_array[..., 2]

    #normalizziamo i valori per stare attorno all'intervallo -0.5, 0.5
    R = R / 255
    R = R - 0.5
    G = G / 255
    G = G - 0.5
    B = B / 255
    B = B - 0.5

    # calcolo filtro mediano
    R_mf = nd.median_filter (R, size =size)
    G_mf = nd.median_filter (G, size = size)
    B_mf = nd.median_filter (B, size = size)


    # edge detection
    # usiamo per il momento i parametri di default e ci riserviamo in futuro la possibilità di 
    # sperimentare con parametri diversi
    R_edge = nd.sobel(R)
    G_edge = nd.sobel(G)
    B_edge = nd.sobel(B)

    # varianza
    mean_R = nd.uniform_filter(R, (size, size))
    mean_G = nd.uniform_filter(G, (size, size))
    mean_B = nd.uniform_filter(B, (size, size))
    mean_sqr_R = nd.uniform_filter(R**2, (size,size))
    mean_sqr_G = nd.uniform_filter(G**2, (size,size))
    mean_sqr_B = nd.uniform_filter(B**2, (size,size))

    R_variance = mean_sqr_R - mean_R**2
    G_variance = mean_sqr_G - mean_G**2
    B_variance = mean_sqr_B - mean_B**2

    #entropia
    R_entropy = local_entropy(R, size)
    G_entropy = local_entropy(G, size)
    B_entropy = local_entropy(B, size)

# ----- SPAZIO COLORE LAB -----
   
    image_array_lab = rgb2lab(image_array)

    #separiamo le componenti L, a, b
    L = image_array_lab[..., 0]
    a = image_array_lab[..., 1]
    b = image_array_lab[..., 2]

    #normalizziamo i valori per stare attorno all'intervallo -0.5, 0.5
    L = L / 100
    L = L - 0.5
    a = a / 50
    b = b / 50

    # calcolo filtro mediano
    
    L_mf = nd.median_filter (L, size =size)
    a_mf = nd.median_filter (a, size = size)
    b_mf = nd.median_filter (b, size = size)


    # edge detection
    # usiamo per il momento i parametri di default e ci riserviamo in futuro la possibilità di 
    # sperimentare con parametri diversi
    L_edge = nd.sobel(L)
    a_edge = nd.sobel(a)
    b_edge = nd.sobel (b)

    
    # varianza
    mean_L = nd.uniform_filter(L, (size, size))
    mean_a = nd.uniform_filter(a, (size, size))
    mean_b = nd.uniform_filter(b, (size, size))
    mean_sqr_L = nd.uniform_filter(L**2, (size,size))
    mean_sqr_a = nd.uniform_filter(a**2, (size,size))
    mean_sqr_b = nd.uniform_filter(b**2, (size,size))

    L_variance = mean_sqr_L - mean_L**2
    a_variance = mean_sqr_a - mean_a**2
    b_variance = mean_sqr_b - mean_b**2

     #entropia
    L_entropy = local_entropy(L, size)
    a_entropy = local_entropy(a, size)
    b_entropy = local_entropy(b, size)

# ----- SPAZIO COLORE HSV -----
   
    image_array_hsv = rgb2hsv(image_array)

    #separiamo le componenti H, S, V
    h = image_array_lab[..., 0]
    s = image_array_lab[..., 1]
    v = image_array_lab[..., 2]

    #normalizziamo i valori per stare attorno all'intervallo -0.5, 0.5
    h = h / 360
    h = h - 0.5
    s = s - 0.5
    v = v - 0.5
   

    # calcolo filtro mediano
    
    h_mf = nd.median_filter (h, size =size)
    s_mf = nd.median_filter (s, size = size)
    v_mf = nd.median_filter (v, size = size)


    # edge detection
    # usiamo per il momento i parametri di default e ci riserviamo in futuro la possibilità di 
    # sperimentare con parametri diversi
    h_edge = nd.sobel(h)
    s_edge = nd.sobel(s)
    v_edge = nd.sobel(v)

    # varianza
    mean_h = nd.uniform_filter(h, (size, size))
    mean_s = nd.uniform_filter(s, (size, size))
    mean_v = nd.uniform_filter(v, (size, size))
    mean_sqr_h = nd.uniform_filter(h**2, (size,size))
    mean_sqr_s = nd.uniform_filter(s**2, (size,size))
    mean_sqr_v = nd.uniform_filter(v**2, (size,size))

    h_variance = mean_sqr_h - mean_h**2
    s_variance = mean_sqr_s - mean_s**2
    v_variance = mean_sqr_v - mean_v**2

     #entropia
    h_entropy = local_entropy(h, size)
    s_entropy = local_entropy(s, size)
    v_entropy = local_entropy(v, size)

    
        

    # Organizziamo i dati per l'utilizzo con algoritmi di machine learning

    layers = [L,a,b, L_mf, a_mf, b_mf, L_edge, a_edge, b_edge, L_variance, a_variance, b_variance, L_entropy, a_variance, b_variance,\
        R, G, B, R_mf, G_mf, B_mf, R_edge, G_edge, B_edge, R_variance, G_variance, B_variance, R_entropy, G_entropy, B_entropy, \
        h, s, v, h_mf, s_mf, v_mf, h_edge, s_edge, v_edge, h_variance, s_variance, v_variance, h_entropy, s_entropy, v_entropy]

    

    # Unisco i vettori di features in una matrice con una riga per featur ed una colonna per pixel
    data= np.stack([l.ravel() for l in layers])

    # Traspongo perchè voglio colonne per features e righe per pixel
    data = data.T
    return data

def pixel_to_class(pixel):
    """
    Assegna ad un singolo pixel una label intera in base al colore di cui è stato colorato
    """
    
    if (pixel == [255,255,255,255]).all(): # pixel del vetrino
        return 0
    elif (pixel == [163,167,69,255]).all(): #strato corneo
        return 1
    elif(pixel == [0,255,255,255]).all(): # derma
        return 2
    elif(pixel == [25, 55, 190, 255]).all(): # pixel dell'epidermide
        return 3
    else: #pixel dei vasi
        return 4


def colimage_to_classes(img):
    """
    Crea le classi per ogni pixel a partire da un'immagine colorata.
    
    img: str, uri dell'immagine colorata.
    """

    img_col = imageio.imread(img)

    classes_matrix = []
    

    for l in range(len(img_col)):
        for c in range (len(img_col[0])):
            classes_matrix.append(pixel_to_class(img_col[l][c]))

    classes_matrix = np.array(classes_matrix)

    return classes_matrix

def class_to_pixel (intClass):
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
    Converte un array di classi y in un'immagine
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