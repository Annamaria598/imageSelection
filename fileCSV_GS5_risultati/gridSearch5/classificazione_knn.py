import imageio
import numpy as np
import scipy.ndimage as nd
import sklearn
from skimage.color import rgb2lab, rgba2rgb, rgb2hsv
from scipy.stats import entropy
from sklearn.metrics import matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

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
    layers['hsv_sin'] = np.sin(h*2*np.pi)*s*v
    layers['hsv_cos'] = np.cos(h*2*np.pi)*s*v
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


def moda(l):
    
    classi = list(set(l))
    
    contatore = {}
    for el in classi:
        contatore[el] = 0
    
    for el in l:
        contatore[el] += 1
        
    keymax = max(contatore, key=contatore.get)
    
    return keymax


def postprocessing_classes(y, shape, window_size=3):
    """
    Effettuiamo il post-processing su un'immagine generata
    assegnando ad ogni pixel la classe più ricorrente nel
    proprio intorno di lato window_size.

    y: classi ottenute dai nostri modelli
    shape: forma dell'immagine originale,
    window_size: dimensione della finestra per il filtro per post-processing.
    """

    y_matrix = y.reshape(shape[:2])

    y_matrix = nd.generic_filter(y_matrix, moda, window_size)

    # srotoliamo la matrice per poter riutilizzare classes_to_colimage
    return y_matrix.ravel()


def select_layers(layers, feature_names=[]):
    """Seleziona dall'insieme di tutte le features solo quelle il cui
    nome compare in feature_names. Se feature_names è vuota o non presente
    vengono selezionate tutte le features in layers."""
    
    if len(feature_names) == 0:
        feature_names = [
            
            "R", "G", "B",
            "R_mf", "G_mf", "B_mf",
            "R_edge", "G_edge", "B_edge",
            "R_variance", "G_variance", "B_variance",

            "L", "a", "b",
            "L_mf", "a_mf", "b_mf",
            "L_edge", "a_edge", "b_edge",
            "L_variance", "a_variance", "b_variance",

            "h", "s", "v",
            "h_mf", "s_mf", "v_mf",
            "h_edge", "s_edge", "v_edge",
            "h_variance", "s_variance", "v_variance",

            "entropy"
        ]
    
    selected = [layers[fn] for fn in feature_names]
    
    # Unisco i vettori di features in una matrice
    # con una riga per featur ed una colonna per pixel
    X = np.stack([l.ravel() for l in selected])

    # Traspongo perchè voglio colonne per features
    # e righe per pixel
    return X.T


def train_on_multi(
    train_images,
    train_images_segmented,
    test_images,
    test_images_segmented,
    window_size = 30,
    neighbors = 5,
    window_size_postprocessing = 3,
    feature_names = []
    ):
    """
    Allena un modello knn per la segmentazione a partire da più
    immagini e lo testa su più immagini.
    Restituisce il modello allenato e lo score raggiunto.

    train_images: lista di nomi di file delle immagini originali;
    train_images_segmented: lista di nomi di file delle immagini segmentate manualmente;
    test_images: lista di nomi di file di immagini originali da usare per il test;
    test_images_segmented: lista di nomi di file di immagini segmentate manualmente da usare per il test;
    window_size: dimensione della finestra da usare per i filtri;
    neighbors: numero di neighbors per l'algoritmo knn;
    window_size_postprocessing: dimensione della finestra per il filtro moda usato come post-processing;
    feature_names: lista di nomi delle features da usare per il modello;
    """

    knn = KNeighborsClassifier(n_neighbors=neighbors)

    X_list = []
    y_list = []
    
    for ti, tis in zip(train_images, train_images_segmented):
        # estrazione delle feature dall'immagine di training
        layers = image_to_data(ti, size=window_size)

        # selezione delle features da utilizzare per questo modello
        X_list.append(select_layers(layers, feature_names))

        # estrazione delle classi dall'immagine segmentata a mano
        y_list.append(colimage_to_classes(tis))

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    # training di un KNN
    knn.fit(X, y)
    
    # test sulle immagini di test
    matthews_score = test_on_multiple_images(knn, test_images, test_images_segmented, window_size, feature_names, window_size_postprocessing)

    return knn, matthews_score


def image_segmentation(
    knn,
    test_image,
    test_image_segmented,
    window_size=30,
    window_size_postprocessing = 3,
    feature_names = []
    ):
    """
    Crea un grafico per rappresentare la qualità di un modello applcandolo
    ad un'immagine di test.

    knn: modello knn allenato,
    test_image: uri di un'immagine originale da usare come test,
    test_image_segmented: uri della versione segmentata dell'immagine di test,
    window_size_postprocessing: dimensione della finestra per il filtro moda usato come postprocessing,
    feature_names: lista dei nomi delle features usate dal modello
    """
    
    # estrazione delle features dall'immagine di test
    test_layers = image_to_data(test_image, size=window_size)
    
    # selezione delle features da utilizzare
    X_test = select_layers(test_layers, feature_names)
    
    # estrazione delle classi dall'immagine di test segmentata a mano
    y_test = colimage_to_classes(test_image_segmented)
    
    # predizione delle classi sull'immagine di test
    y_predette = knn.predict(X_test)
    
    # calcolo dello score per l'immagine segmentata dal modello
    matthews_score = matthews_corrcoef(y_test, y_predette)
    
    # recupero l'immagine di test originale
    test_img = imageio.imread(test_image_segmented)
    
    # generiamo l'immagine ottenuta dal knn
    img_predetta = classes_to_colimage(y_predette, test_img.shape)
    
    # esecuzione post-processing su y ottenute dal knn
    y_postprocessing = postprocessing_classes(y_predette, test_img.shape, window_size_postprocessing)
    
    # generazione dell'immagine dopo il postprocessing
    img_postprocessing = classes_to_colimage(y_postprocessing, test_img.shape)
    
    # calcolo dello score per immagine segmentata dal modello dopo il post-processing
    matthews_score_postprocessing = matthews_corrcoef(y_test, y_postprocessing)
    
    # apertura immagine di test originale
    test_img_orig = imageio.imread(test_image)
    
    # creazione dell'immagine finale
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(test_img_orig)
    axs[0, 0].set_title('Immagine originale')
    axs[0, 1].imshow(test_img)
    axs[0, 1].set_title('Immagine segmentata manualmente')
    axs[1, 0].imshow(img_predetta)
    axs[1, 0].set_title('Immagine segmentata dal modello')
    axs[1, 1].imshow(img_postprocessing)
    axs[1, 1].set_title('Immagine segmentata dopo postprocessing')
    
    # Aggiungo al grafico i matthews coefficients
    max_y = axs[1, 1].get_ylim()[0]
    axs[1, 0].text(0, max_y+70, "Matthews Coeff: {}".format(matthews_score))
    axs[1, 1].text(0, max_y+70, "Matthews Coeff: {}".format(matthews_score_postprocessing))


def test_on_multiple_images(model,
                            test_images,
                            test_images_segmented,
                            window_size=30,
                            feature_names=[],
                            window_size_postprocessing=3):
    """Restituisce il matthews coefficient ottenuto con
    un modello in media su più immagini."""
    
    X_test_matrices = []
    y_test_matrices = []
    
    for test_i, test_i_s in zip(test_images, test_images_segmented):
        # estrazione delle features dall'immagine di test
        test_layers = image_to_data(test_i, size=window_size)

        # selezione delle features da utilizzare
        X_test_matrices.append(select_layers(test_layers, feature_names))

        # estrazione delle classi dall'immagine di test segmentata a mano
        y_test_matrices.append(colimage_to_classes(test_i_s))

    # concatenazione di tutti i dati
    X_test = np.concatenate(X_test_matrices)
    
    y_test = np.concatenate(y_test_matrices)
    
    # predizione delle classi sull'immagine di test
    y_predette = model.predict(X_test)
    
    # calcolo dello score per l'immagine segmentata dal modello
    matthews_score = matthews_corrcoef(y_test, y_predette)
    
    return matthews_score