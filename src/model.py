# src/model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def preparer_et_entrainer_did(gdf_final):
    """
    Entraîne un modèle capable de capturer l'effet de la distance aux stations A et B.
    """
    # 1. Création de variables binaires de proximité (le "Traitement")
    # Considérons une station comme impactante si à moins de 800m
    gdf_final['proche_station_A'] = (gdf_final['dist_metro_A'] < 800).astype(int)
    gdf_final['proche_station_B'] = (gdf_final['dist_metro_B'] < 800).astype(int)
    
    # 2. Préparation des features
    features = [
        'nombre_pieces_principales', 'surface_terrain', 
        'dist_metro_A', 'dist_metro_B', 
        'proche_station_A', 'proche_station_B'
    ]
    
    X = gdf_final[features]
    y = gdf_final['prix_m2']
    
    # 3. Modélisation (RandomForest est excellent pour capter les interactions non-linéaires)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    return model, features

def predire_impact_nouvelle_station(model, nouveau_bien_features):
    """
    nouveau_bien_features: dictionnaire avec les valeurs pour chaque feature
    """
    df_input = pd.DataFrame([nouveau_bien_features])
    # Ajout logique de calcul auto des variables binaires si manquantes
    if 'proche_station_A' not in df_input.columns:
        df_input['proche_station_A'] = (df_input['dist_metro_A'] < 800).astype(int)
        
    return model.predict(df_input)[0]