import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def preparer_et_entrainer_did(gdf_final):
    """
    Entraîne un modèle complet avec Feature Engineering.
    """
    # 1. Feature Engineering (Traitement des données)
    df = gdf_final.copy()
    
    # Conversion date en année (numeric)
    df['annee'] = pd.to_datetime(df['date_mutation'], errors='coerce').dt.year.fillna(2026)
    
    # Encodage du type_local
    le_type = LabelEncoder()
    df['type_local_encoded'] = le_type.fit_transform(df['type_local'].astype(str))
    
    # Remplissage des valeurs manquantes (pour la surface terrain par exemple)
    df['surface_terrain'] = df['surface_terrain'].fillna(0)
    
    # 2. Choix des features (Variables explicatives)
    features = [
        'nombre_pieces_principales', 
        'surface_reelle_bati', 
        'surface_terrain', 
        'dist_metro_A', 
        'dist_metro_B',
        'annee', 
        'type_local_encoded'
    ]
    
    X = df[features]
    y = df['prix_m2']
    
    # 3. Entraînement
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    return model, features, le_type

def predire_impact_nouvelle_station(model, nouveau_bien_features, label_encoder):
    """
    Prédiction robuste avec conversion automatique des types.
    """
    # Conversion du dictionnaire en DataFrame
    df_input = pd.DataFrame([nouveau_bien_features])
    
    # Application du même encodage que lors de l'entraînement
    if 'type_local' in df_input.columns:
        df_input['type_local_encoded'] = label_encoder.transform(df_input['type_local'].astype(str))
        df_input = df_input.drop(columns=['type_local'])
        
    # Liste des colonnes attendues (IMPORTANT : ordre identique au training)
    features_attendues = [
        'nombre_pieces_principales', 'surface_reelle_bati', 'surface_terrain', 
        'dist_metro_A', 'dist_metro_B', 'annee', 'type_local_encoded'
    ]
    
    # Remplissage automatique si une colonne manque
    for col in features_attendues:
        if col not in df_input.columns:
            df_input[col] = 0
            
    # Tri et prédiction
    df_input = df_input[features_attendues]
    return model.predict(df_input)[0]