import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import TwoSlopeNorm


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


def carte_plus_value(
    gdf_final,
    gdf_metro,
    model,
    le,
    ligne,
    distance_nouvelle_station=100,
    figsize=(10, 10)
):
    """
    Affiche une carte de la plus-value simulée si une nouvelle station
    ouvre à `distance_nouvelle_station` mètres de chaque bien.
    """
   
    features_attendues = [
        'nombre_pieces_principales', 'surface_reelle_bati', 'surface_terrain',
        'dist_metro_A', 'dist_metro_B', 'annee', 'type_local_encoded',
    ]

    gdf = gdf_final.copy()
    gdf['annee'] = pd.to_datetime(gdf['date_mutation'], errors='coerce').dt.year.fillna(2022)
    gdf['surface_terrain'] = gdf['surface_terrain'].fillna(0)
    gdf['type_local_encoded'] = le.transform(gdf['type_local'].astype(str))

    X_actuel = gdf[features_attendues].copy()
    prix_actuels = model.predict(X_actuel)

    X_scenario = X_actuel.copy()
    X_scenario[f'dist_metro_{ligne}'] = np.minimum(
        X_actuel[f'dist_metro_{ligne}'],
        distance_nouvelle_station
    )
    prix_scenarios = model.predict(X_scenario)

    masque_affecte = X_actuel[f'dist_metro_{ligne}'] > distance_nouvelle_station
    gdf['plus_value_simulee'] = np.where(
        masque_affecte,
        (prix_scenarios - prix_actuels) / prix_actuels * 100,
        np.nan
    )

    gdf = gdf.to_crs(epsg=4326)
    gdf_metro_wgs = gdf_metro.to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=figsize)

    norm = TwoSlopeNorm(
        vmin=gdf['plus_value_simulee'].quantile(0.05),
        vcenter=0,
        vmax=gdf['plus_value_simulee'].quantile(0.95)
    )

    gdf.plot(
        column='plus_value_simulee',
        ax=ax,
        cmap='RdYlGn',
        norm=norm,
        markersize=8,
        alpha=0.7,
        legend=True,
        legend_kwds={'label': "Plus-value simulée (%)", 'shrink': 0.6},
        missing_kwds={'color': 'lightgrey', 'label': 'Déjà bien desservi'}
    )

    gdf_metro_wgs.plot(
        ax=ax, color='black', markersize=30,
        marker='s', zorder=5, label='Stations existantes'
    )

    ax.set_title(
        f"Plus-value simulée si une station Ligne {ligne} "
        f"ouvre à {distance_nouvelle_station}m de chaque bien",
        fontsize=13
    )
    ax.set_axis_off()
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()