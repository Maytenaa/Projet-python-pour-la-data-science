### IMPACT DE LA MISE EN SERVICE D'UNE NOUVELLE STATION DE METRO SUR LE PRIX DE L'IMMOBILIER
## Projet python pour la data science 2A 
*Sonny Augusto, Abel Cornet-Carlos, Mayténa Labinsky* 


# Objectifs 

L'objectif est de quantifier précisément la plus-value immobilière (en pourcentage) générée par la mise en service d'une nouvelle station de métro du réseau rennais dans le quartier desservi.

# Modèle utilisé


# Sources des données 

Deux bases de données ont été exploitées lors de ce projet. 

Premièrement, le jeu de données [Demandes de valeurs foncières](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres/reuses_and_dataservices) publié et produit par la direction générale des finances publiques, permet de connaître les transactions immobilières intervenues au cours des années 2021 à 2025 sur le territoire métropolitain et les DOM-TOM. Ce jeu de données à été filtré des le début pour ne garder que les villes de Rennes, Cesson-Sévigné et Saint-Jacques-de-lande, c'est à dire les villes desservis par le métro rennais.

Deuxièmement, le jeu de données [Topologie des points d'arrêt de métro du réseau STAR](https://data.rennesmetropole.fr/explore/dataset/topologie-des-points-darret-de-metro-du-reseau-star/api/?location=13,48.10898,-1.6654&basemap=0a029a&dataChart=eyJxdWVyaWVzIjpbeyJjb25maWciOnsiZGF0YXNldCI6InRvcG9sb2dpZS1kZXMtcG9pbnRzLWRhcnJldC1kZS1tZXRyby1kdS1yZXNlYXUtc3RhciIsIm9wdGlvbnMiOnt9fSwiY2hhcnRzIjpbeyJhbGlnbk1vbnRoIjp0cnVlLCJ0eXBlIjoiY29sdW1uIiwiZnVuYyI6IkNPVU5UIiwic2NpZW50aWZpY0Rpc3BsYXkiOnRydWUsImNvbG9yIjoiIzY2YzJhNSJ9XSwieEF4aXMiOiJpZCIsIm1heHBvaW50cyI6NTAsInNvcnQiOiIifV0sInRpbWVzY2FsZSI6IiIsImRpc3BsYXlMZWdlbmQiOnRydWUsImFsaWduTW9udGgiOnRydWV9) contient la liste des points d'arrêt de métro du réseau STAR, le métro rennais,  comprenant notamment leur nom, leur station de rattachement et la géolocalisation.

# Présentation du dépot
Le notebook principal est 

Le dossier src contient tout les modules utilisés dans le notebook blabla.ipynb 

Dans le dossier src, 
- le module get_data.py permet d'extraire les données, 
- le module clean_data.py permet de nettoyer la base de données pour la rendre exploitable, 
- le module stats_desc.py permet de faire de la data visualisation ainsi que de connaître nos données,
- le module model.py permet d'entrainer notre modèle. 

# Navigation au sein du dépot
Il suffit d'executer successivement les cellules du notebook.