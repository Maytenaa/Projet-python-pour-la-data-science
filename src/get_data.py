import os
import pandas as pd
import geopandas as gpd

# Configuration des dossiers
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def fetch_metro_api(url):
    """
    Récupère la topologie des points d'arrêt du métro via l'API Opendatasoft de Rennes.
    On utilise le format GeoJSON pour conserver les géométries.
    """
    print("--- Récupération des données Métro via API (Rennes Métropole) ---")

    # URL de l'API Opendatasoft pour le jeu de données des points d'arrêt
    try:
        # GeoPandas lit directement l'URL d'un flux GeoJSON
        gdf_metro = gpd.read_file(url)
        return gdf_metro
    except Exception as e:
        print(f"Erreur lors de la récupération du métro : {e}")
        return None


def fetch_dvf_api(url):
    """
    Récupère les données DVF via un miroir stable ou le stockage s3 si disponible.
    """
    print("--- Récupération DVF (Source miroir stable) ---")

    try:
        chunks = pd.read_csv(
            url, compression="gzip", sep=",", low_memory=False, chunksize=100000
        )

        df_rennes_list = []
        for chunk in chunks:
            codes_recherche = ["35238", "35051", "35281"]
            filtered_chunk = chunk[
                chunk["code_commune"].astype(str).isin(codes_recherche)
            ].copy()
            df_rennes_list.append(filtered_chunk)

        df_rennes = pd.concat(df_rennes_list)

        return df_rennes
    except Exception as e:
        print(f"Erreur lors de la récupération des données : {e}")
        return None
