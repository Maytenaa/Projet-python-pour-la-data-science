import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def preparer_et_entrainer(gdf_final):
    """
    Entraîne un Random Forest sur gdf_final (sortie de merge_dvf_by_line).
    Retourne le modèle, la liste des features, et le LabelEncoder du type_local.
    """
    df = gdf_final.copy()

    df['annee'] = pd.to_datetime(df['date_mutation'], errors='coerce').dt.year.fillna(2022)
    df['surface_terrain'] = df['surface_terrain'].fillna(0)

    le_type = LabelEncoder()
    df['type_local_encoded'] = le_type.fit_transform(df['type_local'].astype(str))

    features = [
        'nombre_pieces_principales',
        'surface_reelle_bati',
        'surface_terrain',
        'dist_metro_A',
        'dist_metro_B',
        'annee',
        'type_local_encoded',
    ]

    X = df[features].dropna()
    y = df.loc[X.index, 'prix_m2']

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)

    return model, features, le_type


def predire_impact_nouvelle_station(model, nouveau_bien_features, label_encoder):
    """
    Prédit le prix au m² pour un bien décrit par un dictionnaire de features.

    Exemple d'appel :
        bien = {
            'nombre_pieces_principales': 3,
            'surface_reelle_bati': 65,
            'surface_terrain': 0,
            'dist_metro_A': 800,
            'dist_metro_B': 150,   # <- distance à la nouvelle station
            'annee': 2026,
            'type_local': 'Appartement',
        }
        prix = predire_impact_nouvelle_station(model, bien, le)
    """
    df_input = pd.DataFrame([nouveau_bien_features])

    if 'type_local' in df_input.columns:
        df_input['type_local_encoded'] = label_encoder.transform(
            df_input['type_local'].astype(str)
        )
        df_input = df_input.drop(columns=['type_local'])

    features_attendues = [
        'nombre_pieces_principales', 'surface_reelle_bati', 'surface_terrain',
        'dist_metro_A', 'dist_metro_B', 'annee', 'type_local_encoded',
    ]

    for col in features_attendues:
        if col not in df_input.columns:
            df_input[col] = 0

    return model.predict(df_input[features_attendues])[0]