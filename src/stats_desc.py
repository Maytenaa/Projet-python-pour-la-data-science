import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


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
    # 1. On travaille sur une copie pour ne pas modifier l'original par erreur
    gdf_stats = gdf.copy()

    # 2. CRUCIAL : Créer la colonne manquante
    # On calcule pour chaque ligne la distance la plus courte entre la ligne A et la ligne B
    gdf_stats["dist_min_metro"] = gdf_stats[["dist_metro_A", "dist_metro_B"]].min(
        axis=1
    )

    # 3. Définir les tranches et les étiquettes
    bins = [-1, 250, 500, 800, float("inf")]
    labels = ["< 250m", "250m - 500m", "500m - 800m", "> 800m"]

    # 4. Créer la colonne de segments (maintenant dist_min_metro existe !)
    gdf_stats["tranche_distance"] = pd.cut(
        gdf_stats["dist_min_metro"], bins=bins, labels=labels
    )

    # 5. Grouper et calculer la moyenne
    resultat = (
        gdf_stats.groupby("tranche_distance", observed=False)["prix_m2"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Renommer pour le propre
    resultat.columns = [
        "Tranche de distance",
        "Prix moyen au m2 (€)",
        "Nombre de ventes",
    ]

    return resultat


def plot_prix_par_tranche(df_resultat):
    """
    Génère un graphique à barres montrant le prix moyen par tranche de distance.
    """
    plt.figure(figsize=(10, 6))

    # Création du graphique à barres
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(
        data=df_resultat,
        x="Tranche de distance",
        y="Prix moyen au m2 (€)",
        palette="viridis",
    )

    # Ajout des étiquettes de prix au-dessus de chaque barre
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0f} €",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )

    plt.title(
        "Impact de la proximité du métro sur le prix immobilier à Rennes", fontsize=14
    )
    plt.xlabel("Distance à la station la plus proche", fontsize=12)
    plt.ylabel("Prix moyen au m² (€)", fontsize=12)

    # Ajustement des marges pour ne pas couper les étiquettes
    plt.ylim(0, df_resultat["Prix moyen au m2 (€)"].max() * 1.15)

    plt.show()


def compare_proximity_controlled(gdf, variable_controle="type_local"):
    """
    Compare les prix par tranche de distance en neutralisant une variable (ex: type de bien).
    """
    gdf = gdf.copy()
    gdf["dist_min_metro"] = gdf[["dist_metro_A", "dist_metro_B"]].min(axis=1)

    # Création simplifiée Proche vs Loin
    gdf["proximite"] = gdf["dist_min_metro"].apply(
        lambda x: "Proche (<500m)" if x < 500 else "Loin (>500m)"
    )

    # Pivot table pour voir le prix moyen par type de bien ET proximité
    comparaison = gdf.pivot_table(
        values="prix_m2", index=variable_controle, columns="proximite", aggfunc="mean"
    )

    # Calcul de la plus-value en %
    comparaison["Plus-value (%)"] = (
        (comparaison["Proche (<500m)"] / comparaison["Loin (>500m)"]) - 1
    ) * 100

    return comparaison


# DiD


def prepare_did_data(gdf):
    df_did = gdf.copy()

    # 1. Calcul de la distance minimale au métro (A ou B)
    # On crée une colonne temporaire pour simplifier la condition
    dist_min = gdf[["dist_metro_A", "dist_metro_B"]].min(axis=1)

    # 2. Définir le groupe Traité (< 250m de n'importe quelle ligne)
    df_did["treated"] = (dist_min <= 250).astype(int)

    # 3. Définir la période Post-ouverture (Ligne B : 20 Septembre 2022)
    # Attention : comme vu précédemment, assure-toi d'avoir des données avant cette date !
    df_did["post_event"] = (df_did["date_mutation"] >= "2022-09-20").astype(int)

    return df_did


def run_did_regression(df_did):
    # Modèle avec contrôles pour neutraliser les autres effets
    # On ajoute le type_local et les pièces pour comparer des choses comparables
    model = smf.ols(
        formula="prix_m2 ~ treated * post_event + type_local + nombre_pieces_principales",
        data=df_did,
    ).fit()

    return model


def plot_did_trends(df_did):
    # On crée une colonne Année-Mois pour voir l'évolution
    df_did["period"] = df_did["date_mutation"].dt.to_period(
        "Q"
    )  # Par trimestre pour plus de lissé

    # Calcul des moyennes par période et par groupe
    trends = df_did.groupby(["period", "treated"])["prix_m2"].mean().unstack()

    # Plot
    trends.plot(figsize=(12, 6), marker="o")
    plt.axvline(
        x=pd.Period("2022Q3"), color="red", linestyle="--", label="Ouverture Ligne B"
    )
    plt.title("Évolution des prix : Groupe Traité vs Groupe Contrôle")
    plt.ylabel("Prix moyen au m² (€)")
    plt.legend(["Contrôle (Loin)", "Traité (Ligne B)"])
    plt.grid(True, alpha=0.3)
    plt.show()



