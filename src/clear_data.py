import geopandas as gpd
import pandas as pd


def merge_yearly_dvf(df_list):
    """
    Fusionne une liste de DataFrames DVF en un seul.
    """
    df_merged = pd.concat(df_list, ignore_index=True)
    return df_merged


def clean_dvf_data(df_raw):
    """
    Nettoie les données DVF : filtrage Rennes, conversion GDF, projection 2154.
    """
    df = df_raw.copy()

    # 1. Filtrage sur les communes contenant un arrêt de métro
    # On définit la liste des codes communes souhaités
    codes_recherche = ["35238", "35051", "35281"]
    df = df[df["code_commune"].astype(str).isin(codes_recherche)]

    # 2. Filtrage métier : on ne garde que les ventes de maisons/appartements
    df = df[df["nature_mutation"] == "Vente"]
    df = df[df["type_local"].isin(["Appartement", "Maison"])]

    # 3. Suppression des lignes sans prix ou sans coordonnées
    df = df.dropna(
        subset=["valeur_fonciere", "latitude", "longitude", "surface_reelle_bati"]
    )

    # 4. Conversion en GeoDataFrame (WGS84 d'abord)
    gdf_dvf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )

    # 5. Projection en Lambert 93 (mètres) pour les calculs de distance
    gdf_dvf = gdf_dvf.to_crs(epsg=2154)

    # 6. Nettoyage statistique
    gdf_dvf = gdf_dvf[gdf_dvf["surface_reelle_bati"] > 0]

    # Sélection des colonnes essentielles pour l'analyse de prix
    colonnes_essentielles = [
        "valeur_fonciere",
        "date_mutation",
        "nature_mutation",
        "type_local",
        "surface_reelle_bati",
        "nombre_pieces_principales",
        "surface_terrain",
        "code_commune",
        "nom_commune",
        "latitude",
        "longitude",
        "geometry",
    ]

    # On ne garde que les colonnes qui existent réellement dans le dataframe
    gdf_dvf = gdf_dvf[gdf_dvf.columns.intersection(colonnes_essentielles)]
    return gdf_dvf


STATIONS_A = [
    "J.F. Kennedy",
    "Villejean-Université",
    "Pontchaillou",
    "Anatole France",
    "Sainte-Anne",
    "République",
    "Charles de Gaulle",
    "Gares",
    "Jacques Cartier",
    "Clemenceau",
    "Henri Fréville",
    "Italie",
    "Triangle",
    "Le Blosne",
    "La Poterie",
]

STATIONS_B = [
    "Saint-Jacques - Gaîté",
    "La Courrouze",
    "Cleunay",
    "Mabilais",
    "Colombier",
    "Gares",
    "Saint-Germain",
    "Sainte-Anne",
    "Jules Ferry",
    "Gros-Chêne",
    "Les Gayeulles",
    "Joliot-Curie - Chateaubriand",
    "Beaulieu - Université",
    "Atalante",
    "Cesson - Viasilva",
]


def clean_metro_data(gdf_metro_raw):
    """
    Nettoie et standardise les données des stations de métro.
    """
    # 1. Copie pour éviter de modifier l'original
    gdf = gdf_metro_raw.copy()

    # 2. Projection en Lambert 93 (EPSG:2154) pour les calculs en mètres
    if gdf.crs != "EPSG:2154":
        gdf = gdf.to_crs(epsg=2154)

    # 3. Standardisation des noms
    gdf["nom"] = gdf["nom"].str.strip()

    # 4. Gestion des doublons géographiques
    gdf = gdf.dissolve(by="nom").reset_index()

    # 5. Sélection des colonnes essentielles
    cols_to_keep = ["nom", "geometry"]
    gdf = gdf[cols_to_keep]

    return gdf


def remove_extreme_values(gdf, variable, seuil_bas, seuil_haut):
    """
    Filtre le GeoDataFrame en fonction des seuils déterminés lors de l'analyse.
    """
    # Application des filtres
    mask = (gdf[f"{variable}"] >= seuil_bas) & (gdf[f"{variable}"] <= seuil_haut)
    # Filtrage final
    df_filtered = gdf[mask].copy()

    return df_filtered


def merge_dvf_by_line(gdf_dvf, gdf_metro):
    """
    Calcule la distance à la ligne A et B
    """

    # 1. Création de deux GeoDataFrames distincts pour les lignes
    metro_a = gdf_metro[gdf_metro["nom"].isin(STATIONS_A)].copy()
    metro_b = gdf_metro[gdf_metro["nom"].isin(STATIONS_B)].copy()

    # 2. Jointure Ligne A
    gdf_res = gpd.sjoin_nearest(
        gdf_dvf, metro_a[["nom", "geometry"]], distance_col="dist_metro_A", how="left"
    )
    gdf_res = gdf_res.rename(columns={"nom": "station_A"})
    gdf_res = gdf_res.drop(columns="index_right", errors="ignore")

    # 3. Jointure Ligne B
    gdf_res = gpd.sjoin_nearest(
        gdf_res, metro_b[["nom", "geometry"]], distance_col="dist_metro_B", how="left"
    )
    gdf_res = gdf_res.rename(columns={"nom": "station_B"})
    gdf_res = gdf_res.drop(columns="index_right", errors="ignore")

    return gdf_res
