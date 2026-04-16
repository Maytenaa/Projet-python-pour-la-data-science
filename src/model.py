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
        'proche_station_A', 'proche_station_B', 
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



def preparer_et_entrainer_did2(gdf_final):
    """
    Entraîne un modèle enrichi avec des variables temporelles et contextuelles.
    """
    # 1. Feature Engineering
    # Extraction de l'année depuis la date_mutation
    gdf_final['annee'] = pd.to_datetime(gdf_final['date_mutation']).dt.year
    
    # Encodage du type de local (ex: Appartement vs Maison)
    le = LabelEncoder()
    gdf_final['type_local_code'] = le.fit_transform(gdf_final['type_local'].astype(str))
    
    # 2. Choix des features pertinentes
    features = [
        'nombre_pieces_principales', 'surface_terrain', 
        'dist_metro_A', 'dist_metro_B',
        'annee', 'type_local_code'
    ]
    
    X = gdf_final[features]
    y = gdf_final['prix_m2']
    
    # 3. Entraînement
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    return model, features, le

def predire_impact_nouvelle_station2(model, nouveau_bien_features, label_encoder):
   
    df_input = pd.DataFrame([nouveau_bien_features])
    
    # S'assurer que le type_local est encodé correctement
    if 'type_local' in df_input.columns:
        df_input['type_local_code'] = label_encoder.transform(df_input['type_local'].astype(str))
        df_input = df_input.drop(columns=['type_local'])
    
    # Ordre des colonnes attendues
    expected_features = ['nombre_pieces_principales', 'surface_terrain', 
                         'dist_metro_A', 'dist_metro_B', 'annee', 'type_local_code']
    
    # Remplissage par défaut (0) si une variable est manquante
    for col in expected_features:
        if col not in df_input.columns:
            df_input[col] = 0
            
    df_input = df_input[expected_features]
    return model.predict(df_input)[0]




































































































































































































































