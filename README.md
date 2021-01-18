# Classificazione con KNN

Nel notebook KNN_prova.ipynb è presente il training di
un singolo modello e la generazione delle immagini da
esso prodotte con e senza post-processing confrontate
all'immagine di test originale.

Nei vari file GridSearchX.py (con X valore numerico) è
presente la grid search da eseguire in parallelo con
più processi. Il numero rappresenta il tentativo di
ricerca di features e parametri.

I risultati delle grid search sono disponibili in formato
csv e sono nominati nella forma results_gsX.csv con X
valore numerico corrispondente alla grid search.

Tutte le funzioni di supporto sono disponibili nel file
classificazione_knn.py.


# ImageSelection

Questo è uno strumento per separare porzioni di immagini 
evidenziate con colori diversi ed espostarle in immagini 
separate.

Tutte le funzioni principali sono nel file core.py.

Esempio di utilizzo disponibile nel file usage.ipynb.