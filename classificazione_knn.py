import imageio 
import numpy as np
import scipy.ndimage as nd 
import sklearn
from skimage.color import rgb2lab, rgba2rgb, rgb2hsv


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
    # in attesa di chiarimenti

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

    
        

    # Organizziamo i dati per l'utilizzo con algoritmi di machine learning

    layers = [L,a,b, L_mf, a_mf, b_mf, L_edge, a_edge, b_edge, \
        R, G, B, R_mf, G_mf, B_mf, R_edge, G_edge, B_edge, \
        h, s, v, h_mf, s_mf, v_mf, h_edge, s_edge, v_edge]

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