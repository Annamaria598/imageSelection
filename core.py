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
        if "." in f and f.split()[1] in formatiAccettati:
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
 
 

 
 

 
 



 
 
def main():
 
    # caricare immagini (colorata e originale)
    imgList = load_images("./img")
 
    # separare porzioni colorate diversamente
   
 
    # esportare le diverse immagini delle porzioni
  
 
   
 
if __name__ == "__main__":
    main()