import os
import pandas as pd
import geopandas as gpd
import requests

# Configuration des dossiers
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_metro_api():
    """
    Récupère la topologie des points d'arrêt du métro via l'API Opendatasoft de Rennes.
    On utilise le format GeoJSON pour conserver les géométries.
    """
    print("--- Récupération des données Métro via API (Rennes Métropole) ---")
    
    # URL de l'API Opendatasoft pour le jeu de données des points d'arrêt
    dataset_id = "topologie-des-points-darret-de-metro-du-reseau-star"
    url = f"https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/{dataset_id}/exports/geojson"
    
    try:
        # GeoPandas lit directement l'URL d'un flux GeoJSON
        gdf_metro = gpd.read_file(url)
        print(f"Succès : {len(gdf_metro)} points d'arrêt de métro récupérés.")
        return gdf_metro
    except Exception as e:
        print(f"Erreur lors de la récupération du métro : {e}")
        return None

def fetch_dvf_api(code_commune="35238"):
    pass

def main():
    # 1. Récupération et sauvegarde du Métro
    gdf_metro = fetch_metro_api()
    if gdf_metro is not None:
        metro_path = os.path.join(DATA_DIR, "metro_stations.geojson")
        gdf_metro.to_file(metro_path, driver='GeoJSON')
        print(f"Fichier métro sauvegardé : {metro_path}")

    # 2. Récupération et sauvegarde des DVF
    df_dvf = fetch_dvf_api()
    if df_dvf is not None:
        dvf_path = os.path.join(DATA_DIR, "dvf_rennes_raw.csv")
        df_dvf.to_csv(dvf_path, index=False)
        print(f"Fichier DVF sauvegardé : {dvf_path}")

if __name__ == "__main__":
    main()