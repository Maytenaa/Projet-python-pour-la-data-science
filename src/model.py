# model.py
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def preparer_et_entrainer(df_rennes, gdf_metro):
    """
    Prépare les données, calcule la distance au métro et entraîne le modèle.
    """
    # 1. Conversion spatiale
    gdf_rennes = gpd.GeoDataFrame(
        df_rennes, 
        geometry=gpd.points_from_xy(df_rennes.longitude, df_rennes.latitude),
        crs="EPSG:4326"
    ).to_crs("EPSG:2154")
    
    gdf_metro = gdf_metro.to_crs("EPSG:2154")

    # 2. Calcul de la distance minimale
    print("Calcul des distances (cela peut prendre un instant)...")
    gdf_rennes['dist_min_metro'] = gdf_rennes.geometry.apply(
        lambda geom: gdf_metro.distance(geom).min()
    )

    # 3. Préparation des features
    gdf_rennes['date_mutation'] = pd.to_datetime(gdf_rennes['date_mutation'])
    gdf_rennes['annee'] = gdf_rennes['date_mutation'].dt.year
    
    df_clean = gdf_rennes.dropna(subset=['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales'])
    
    le = LabelEncoder()
    df_clean['type_local_code'] = le.fit_transform(df_clean['type_local'].astype(str))

    features = ['surface_reelle_bati', 'nombre_pieces_principales', 'dist_min_metro', 'annee', 'type_local_code']
    X = df_clean[features]
    y = df_clean['valeur_fonciere']

    # 4. Entraînement
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, features


def predire_impact_nouvelle_station(model, surface, pieces, type_local_code, annee, distance_metro=0):
    """
    Simule le prix d'un bien avec une nouvelle station de métro (distance=0).
    """
    # Création du dataframe d'input avec les mêmes colonnes que lors de l'entraînement
    input_data = pd.DataFrame({
        'surface_reelle_bati': [surface],
        'nombre_pieces_principales': [pieces],
        'dist_min_metro': [distance_metro], # 0 pour une station juste à côté
        'annee': [annee],
        'type_local_code': [type_local_code]
    })
    
    return model.predict(input_data)[0]