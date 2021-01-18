import numpy as np
import os
import imageio

def load_images(directory):
    """
    directory: str, la directory in cui cercare le immagini.

    Riceve una directory, e restituisce una lista di tuple
    immagine originale-immagine colorata come matrici numpy.
    """

    # leggere lista file dentro directory
    fileNamesList = os.listdir(directory)

    # filtrare soltanto file immagine png
    formatiAccettati = ["png", "PNG"]

    fileNamesList = [ f for f in fileNamesList if "." in f and f.split(".")[1] in formatiAccettati ]

    # raccogliamo soltanto i nomi non colors
    fileNamesList = [ f for f in fileNamesList if "_" not in f ]

    # carico tutte le immagini colorate e non colorate
    coppieImmagini = []

    for fn in fileNamesList:
        thisOI = imageio.imread(directory + "/" + fn)

        colorFn = fn.split(".")[0] + "_colors." + fn.split(".")[1]

        thisCI = imageio.imread(directory + "/" + colorFn)

        coppieImmagini.append( (thisOI, thisCI, fn) )

    return coppieImmagini


def find_portions(colorsDict, imgList):
    """
    Prende come parametri un dizionario con codici RGB
    come chiavi ed etichetta della porzione di quel colore
    come valore ed una lista di immagini come caricate dalla
    funzione load_images.

    Restituisce una nuova lista contenente le immagini originali
    e le immagini corrispondenti con le singole porzioni.
    """

    # raccolgo colori possibili da colorsDict
    colors = list(colorsDict.keys())

    newImgList = []

    for imgTuple in imgList:

        singleImageDict = {}
        singleImageDict["orig"] = imgTuple[0]
        singleImageDict["colored"] = imgTuple[1]
        singleImageDict["name"] = imgTuple[2]

        for color in colors:

            mask = make_mask_by_color(color, singleImageDict["colored"])

            singleImageDict[colorsDict[color]] = split_image_by_color(mask, singleImageDict["orig"])
    
        newImgList.append(singleImageDict)

    return newImgList


def make_mask_by_color(color, image):
    """
    Crea una maschera booleana per selezionare i punti
    dell'immagine in cui il colore è "color"
    """
    # esempio colore: "00ffff"

    r_color = int(color[:2], 16) # comp rosso es: 0
    g_color = int(color[2:4], 16) # comp verde es: 255
    b_color = int(color[4:], 16) # comp blu es: 255

    mask_r = image[:, :, 0] == r_color # punti con mia comp rossa
    mask_g = image[:, :, 1] == g_color # punti con mia comp verde
    mask_b = image[:, :, 2] == b_color # punti con mia comp blu

    mask = np.logical_and(mask_r, mask_g)
    mask = np.logical_and(mask, mask_b)

    return mask


def split_image_by_color(mask, img):
    """
    Restituisce una copia dell'immagine fornita in cui
    le parti non individuate dalla maschera sono rese
    trasparenti.
    """

    newImg = img.copy() # copio immagine per non
                        # modificare l'originale

    mask[:,:] = (mask == False) # inverto i valori bool nella maschera

    newImg[mask] = [0, 0, 0, 0] # setto i pixel non interessanti
                                # a nero trasparente

    return newImg


def save_images(imageList, dir="."):
    """
    Salva tutte le immagini parziali contenute nella
    imageList fornita come parametro nella directory
    dir.
    Di default dir è la directory corrente.
    """

    if not os.path.isdir(dir):
        os.mkdir(dir)

    for i in imageList:
        chiavi = list(i.keys())
        chiavi.remove("orig")
        chiavi.remove("colored")
        chiavi.remove("name")

        name_start = i["name"].split(".")[0]
        extension = i["name"].split(".")[1]

        for k in chiavi:
            imageio.imsave(dir + "/" + name_start + "_" + k + "." + extension, i[k])


def main():

    # caricare immagini (colorata e originale)
    imgList = load_images("./img")

    # separare porzioni colorate diversamente
    dictColori = {
        "ffffff": "vetrino",
        "a3a745": "strato_corneo",
        "00ffff": "epidermide",
        "1937be": "derma"
    }

    imgPortionsList = find_portions(dictColori, imgList)

    # esportare le diverse immagini delle porzioni
    save_images(imgPortionsList, "img/porzioni")

    return imgPortionsList


if __name__ == "__main__":
    main()