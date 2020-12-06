import imageio 
import numpy as np
import scipy.ndimage as nd 
import sklearn
from skimage.color import rgb2lab, rgba2rgb


def image_to_data(img):
    """
    Genera e associa attributi ad ogni pixel dell'immagine utilizzando vari filtri.

    img: str, il path ad un'immagine.
    """

    # apriamo immagine come matrice in RGB
    image_array = imageio.imread(img)

    #convertiamo in RGB in LAB
    # se l'immagine aperta era un PNG abbiamo rgba
    # invece di rgb ma rgb2lab vuole solo i tre canali
    # del colore

    if(img.split(".")[-1] in ["png", "PNG"]):
        image_array = rgba2rgb(image_array)

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
    # scegliamo per iniziare una finestra di dimensione 13, ma nulla ci impedisce più avanti di 
    # sperimentare con valori differenti.
    size = 13
    L_mf_13 = nd.median_filter (L, size =size)
    a_mf_13 = nd.median_filter (a, size = size)
    b_mf_13 = nd.median_filter (b, size = size)


    # edge detection
    # usiamo per il momento i parametri di default e ci riserviamo in futuro la possibilità di 
    # sperimentare con parametri diversi
    L_edge = nd.sobel(L)
    a_edge = nd.sobel(a)
    b_edge = nd.sobel (b)

    # covarianza
    # in attesa di chiarimenti

    
    features_dict = {
        "img": image_array_lab,
        "L" : L,
        "a" : a,
        "b" : b,
        "L_mf": L_mf_13,
        "a_mf": a_mf_13,
        "b_mf": b_mf_13,
        "L_edge" : L_edge,
        "a_edge" : a_edge,
        "b_edge" : b_edge
    }
    

    # Organizziamo i dati per l'utilizzo con algoritmi di machine learning

    layers = [L,a,b, L_mf_13, a_mf_13, b_mf_13, L_edge, a_edge, b_edge]

    # Unisco i vettori di features in una matrice con una riga per featur ed una colonna per pixel
    data= np.stack([l.ravel() for l in layers])

    # Traspongo perchè voglio colonne per features e righe per pixel
    data = data.T
    return data