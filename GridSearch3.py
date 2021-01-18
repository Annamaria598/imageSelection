#!/usr/bin/env python
# coding: utf-8

from classificazione_knn import image_to_data, colimage_to_classes, local_entropy, classes_to_colimage
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef
import numpy as np
import imageio
import matplotlib.pyplot as plt
import csv
import time
from itertools import combinations, chain

# parametri per generazione features
# per ogni combinazione possibile bisogna generare
# le features. Per il momento solo la window size
# è un parametro.
window_size = [20, 25, 30, 35, 40]

# funzione per rimuovere le combinazioni con
# tutte le features provenienti dallo stesso
# spazio colore
def stesso_spazio(features):
    
    lab = ["L", "a", "b"]
    rgb = ["R", "G", "B"]
    hsv = ["h", "s", "v"]
    
    in_lab = 0
    in_rgb = 0
    in_hsv = 0
    
    for f in features:
        element = f.split("_")[0]
        if element in lab:
            in_lab = 1
        elif element in rgb:
            in_rgb = 1
        else:
            in_hsv = 1
            
        if in_lab + in_rgb + in_hsv > 1:
            return False
            
    return True


features_to_mix = [
    ['R_mf', 'G_mf', 'B_mf'],
    ['R_variance', 'G_variance', 'B_variance'],
    ['L_mf', 'a_mf', 'b_mf'],
    ['L_variance', 'a_variance', 'b_variance'],
    ['h_mf', 's_mf', 'v_mf'],
    ['h_variance', 's_variance', 'v_variance'],
    ['hsv_sin','hsv_cos']
    
]

comb2 = combinations(features_to_mix, 2)
comb3 = combinations(features_to_mix, 3)

# appiattiamo le liste di tuple di liste
comb2_lists = []
for c in comb2:
    comb2_lists.append(list(chain.from_iterable(c)))

comb3_lists = []
for c in comb3:
    comb3_lists.append(list(chain.from_iterable(c)))

comb = comb2_lists + comb3_lists

# rimuovo set formati da features da un unico spazio
comb = [c for c in comb if not stesso_spazio(c)]

feature_set = [
    # prime 10 dalla prima grid search
    ['h', 's', 'v', 'h_mf', 's_mf', 'v_mf', 'h_variance', 's_variance', 'v_variance', 'entropy'],
    ['h', 's', 'v', 'h_mf', 's_mf', 'v_mf', 'h_variance', 's_variance', 'v_variance', 'entropy'],
    ['h', 's', 'v', 'h_mf', 's_mf', 'v_mf', 'h_variance', 's_variance', 'v_variance', 'entropy'],
    ['R', 'G', 'B', 'R_mf', 'G_mf', 'B_mf', 'R_variance', 'G_variance', 'B_variance', 'entropy'],
    ['R', 'G', 'B', 'R_mf', 'G_mf', 'B_mf', 'R_variance', 'G_variance', 'B_variance', 'entropy'],
    ['R', 'G', 'B', 'R_mf', 'G_mf', 'B_mf', 'R_variance', 'G_variance', 'B_variance', 'entropy'],
    ['h', 's', 'v', 'h_mf', 's_mf', 'v_mf', 'h_variance', 's_variance', 'v_variance', 'entropy'],
    ['h', 's', 'v', 'h_mf', 's_mf', 'v_mf', 'entropy'],
    ['R', 'G', 'B', 'R_mf', 'G_mf', 'B_mf', 'entropy'],
    ['h', 's', 'v', 'h_mf', 's_mf', 'v_mf', 'entropy']
]

# aggiunta nuovi insiemi di features
feature_set = feature_set + comb

# parametri per il knn
neighbors = [3,5,7,10,15,20,25,50]

# Generazione dei layers per tutti i parametri riguardanti i dati
print("Generazione dei dati...")
for s in window_size:
    # training data
    data = image_to_data("img/pelle303R.PNG", size=s)
    
    for key, value in data.items():
        file_name = "data_ws" + str(s) + "_" + key + ".csv"
        with open("data/" + file_name, "w") as f:
            csvw = csv.writer(f)
            csvw.writerows(value)
            
    # test data
    data = image_to_data("img/pelle305R.PNG", size=s)
    
    for key, value in data.items():
        file_name = "test_data_ws" + str(s) + "_" + key + ".csv"
        with open("data/" + file_name, "w") as f:
            csvw = csv.writer(f)
            csvw.writerows(value)

# generazione delle classi per i pixel
y = colimage_to_classes("img/pelle303R_colors.PNG")

# classi dei pixel per l'immagine di test
y_test = colimage_to_classes("img/pelle305R_colors.PNG")
print("completa.")

# Generazione dei jobs per i modelli da allenare
print("generazione jobs da eseguire...")
from itertools import product

jobs = list(product(window_size, feature_set, neighbors))
print("Completa. {} jobs da eseguire".format(len(jobs)))

from multiprocessing import Pool

def train_model(job):
    
    # istanzio modello da allenare
    knn = KNeighborsClassifier(job[2])
    
    # raccolgo i layers/features che mi interessano dai file csv
    features = []
    for feature in job[1]:
        with open("data/data_ws{}_{}.csv".format(job[0], feature)) as f:
            csvr = csv.reader(f)
            this_feature = []
            for row in csvr:
                this_feature.append(np.array(row, dtype=np.float64))
            features.append(this_feature)
    
    features = np.array(features)
    
    # Unisco i vettori di features in una matrice
    # con una riga per feature ed una colonna per pixel
    X = np.stack([l.ravel() for l in features])

    # Traspongo perchè voglio colonne per features
    # e righe per pixel
    X = X.T
    
    # inizio il training
    knn.fit(X, y)
    
    # score con dati dell'immagine di test
    # raccolgo i layers/features che mi interessano dai file csv
    features = []
    for feature in job[1]:
        with open("data/test_data_ws{}_{}.csv".format(job[0], feature)) as f:
            csvr = csv.reader(f)
            this_feature = []
            for row in csvr:
                this_feature.append(np.array(row, dtype=np.float64))
            features.append(this_feature)
    
    features = np.array(features)
    
    # Unisco i vettori di features in una matrice
    # con una riga per feature ed una colonna per pixel
    X_test = np.stack([l.ravel() for l in features])

    # Traspongo perchè voglio colonne per features
    # e righe per pixel
    X_test = X_test.T
    
    y_predette = knn.predict(X_test)
    
    score = matthews_corrcoef(y_test, y_predette)
    
    return score, job
    

if __name__ == '__main__':
    print("{} : Inizio training".format(time.ctime()))
    with Pool(5) as p:
        results = p.map(train_model, jobs)
        
        with open("results.txt", "w") as f:
            f.write("score, window_size, features, neighbors\n")
            for r in results:
                f.write(str(r[0]) + ", " + str(r[1][0]) + ", \"" + str(r[1][1]) + "\", " + str(r[1][2]) + "\n")
                
        print("{} : Training completo".format(time.ctime()))



