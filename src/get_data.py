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
        return gdf_metro
    except Exception as e:
        print(f"Erreur lors de la récupération du métro : {e}")
        return None

def fetch_dvf_api():
    """
    Récupère les données DVF via un miroir stable ou le stockage s3 si disponible.
    """
    print("--- Récupération DVF (Source miroir stable) ---")
    
    # URL vers un export consolidé spécifique (millésime 2023 complet)
    url = "https://files.data.gouv.fr/geo-dvf/latest/csv/2023/full.csv.gz"
    
    try:
        chunks = pd.read_csv(url, compression='gzip', sep=',', low_memory=False, chunksize=100000)
        
        df_rennes_list = []
        for chunk in chunks:
            filtered_chunk = chunk[chunk['code_commune'].astype(str).str.contains('35238')].copy()
            df_rennes_list.append(filtered_chunk)
            
        df_rennes = pd.concat(df_rennes_list)

        return df_rennes
    except Exception as e:
        print(f"Erreur lors de la récupération des données : {e}")
        return None

def main():
    gdf_metro = fetch_metro_api()
    if gdf_metro is not None:
        metro_path = os.path.join(DATA_DIR, "metro_stations.geojson")
        gdf_metro.to_file(metro_path, driver='GeoJSON')

    df_dvf = fetch_dvf_api()
    if df_dvf is not None:
        dvf_path = os.path.join(DATA_DIR, "dvf_rennes_raw.csv")
        df_dvf.to_csv(dvf_path, index=False)
if __name__ == "__main__":
    main()