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


def analyse_prix_dist_corr(gdf):
    """
    Calcule la corrélation entre le prix au m2 et la distance au métro.
    """
    # On prend la distance au métro le plus proche (A ou B)
    gdf["dist_min_metro"] = gdf[["dist_metro_A", "dist_metro_B"]].min(axis=1)

    correlation = gdf["prix_m2"].corr(gdf["dist_min_metro"])
    return correlation
