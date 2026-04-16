import pandas as pd


def get_general_stats(gdf):
    """
    Retourne les statistiques descriptives globales du prix au m2.
    """
    stats = gdf["prix_m2"].describe()
    return stats


def get_stats_by_ligne(gdf):
    """
    Compare les prix moyens entre les zones proches de la ligne A et de la ligne B.
    (On considère ici la ligne la plus proche pour chaque bien).
    """
    # On crée une colonne pour savoir quelle ligne est la plus proche
    gdf["ligne_proche"] = "A"
    gdf.loc[gdf["dist_metro_B"] < gdf["dist_metro_A"], "ligne_proche"] = "B"

    return gdf.groupby("ligne_proche")["prix_m2"].agg(
        ["mean", "median", "std", "count"]
    )


def analyse_prix_dist_tranche(gdf):
    """
    Calcule le prix moyen au m2 selon des tranches de distance au métro.
    """

    # Définir les tranches et les étiquettes
    # On commence à -1 pour inclure le 0, et float('inf') pour "plus de 800m"
    bins = [-1, 250, 500, 800, float("inf")]
    labels = ["< 250m", "250m - 500m", "500m - 800m", "> 800m"]

    # Créer la colonne de segments
    gdf["tranche_distance"] = pd.cut(gdf["dist_min_metro"], bins=bins, labels=labels)

    # Grouper et calculer la moyenne (et le compte pour vérification)
    resultat = (
        gdf.groupby("tranche_distance", observed=False)["prix_m2"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Renommer pour plus de clarté
    resultat.columns = [
        "Tranche de distance",
        "Prix moyen au m2 (€)",
        "Nombre de ventes",
    ]

    return resultat
