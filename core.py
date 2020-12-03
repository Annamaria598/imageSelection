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
 
    # lista dei nomi delle immagini
    imgNamesList = []

    for f in fileNamesList:
        # se nel nome c'è un punto e dopo il punto png o PNG
        if "." in f and f.split(".")[1] in formatiAccettati:
            imgNamesList.append(f)

    nonColImgNameList = []
    # raccogliamo soltanto i nomi non colors
    
    for f in imgNamesList:
        if "_" not in f:
            nonColImgNameList.append(f)

    # carico tutte le immagini colorate e non colorate
    coppieImmagini = []
 
    for fn in nonColImgNameList:
        # immagine originale
        thisOI = imageio.imread(directory + "/" + fn)

        # immagine colorata
        colorFn = fn.split(".")[0] + "_colors." + fn.split(".")[1]
 
        thisCI = imageio.imread(directory + "/" + colorFn)

        #aggiungo alla lista da restituire
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
            # creo la maschera booleana del colore corrente 
            mask = make_mask_by_color(color, singleImageDict["colored"])

            #raccolgo la funzione per l'immagine individuata dal colore corrente 
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
    
    # creo la maschera per i punti in cui ho True per tutte le maschere delle componenti colore
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
 
    mask[:,:] = mask == False # inverto i valori bool nella maschera
 
    newImg[mask] = [0, 0, 0, 0] # setto i pixel non interessanti
                                # a nero trasparente
 
    return newImg 

def save_images(imageList, dir= "."):
    """
    Salva tutte le immagini parziali contenute nella imageListfornita come parametro nella directory dir.
    Di default dir è la directory corrente.
    """

    # se la directory non esiste la creo
    if not os.path.isdir(dir):
        os.mkdir(dir) 

    
    # salvo le diverse correzioni di immagini chaimandole con le etichette assegnate all'inizio
    for i in imageList:


        # voglio salvare per le diverse porzioni come immagini separae e chiamarle con il nome della porzione
        chiavi = list(i.keys())

        # tolgo le chiavi che non sono immagini da salvare
        chiavi.remove("orig")
        chiavi.remove("colored")
        chiavi.remove("name")

        #raccolgo il nome originale senza estensione
        name_start = i["name"].split(".")[0]

        #raccolgo l'estensione
        extension = i["name"].split(".")[1]


        for k in chiavi:

            # costruisco nome dell'immagine da salvare
            new_name = name_start + "_" + k + "." + extension
            # salvo la singola immagine
            imageio.imsave(dir + "/" + new_name, i[k])
 



 
 
def main():
 
    # caricare immagini (colorata e originale)
    imgList = load_images("./img")
 
    

    # Definisco un dizionario dei colori presenti nell'immagine colorate con etichette per le rispettive 
    # sezioni.
    dictColori = {
        "ffffff": "vetrino",
        "a3a745": "strato_corneo",
        "00ffff": "epidermide",
        "1937be": "derma"
    }
    # separare porzioni colorate diversamente
    imgPortionsList = find_portions(dictColori, imgList)

    # esportare le diverse immagini delle porzioni
    save_images(imgPortionsList,"img/porzioni")
 
   
 
if __name__ == "__main__":
    main()