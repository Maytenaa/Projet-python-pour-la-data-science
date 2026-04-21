import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def extreme_value_prix(df):
    """
    Calcule les seuils bas et hauts basés sur les données réelles du prix au m².
    """
    # Calcul temporaire du prix m2
    temp_p2 = df["valeur_fonciere"] / df["surface_reelle_bati"]
    temp_p2 = temp_p2.dropna()
    Q1 = temp_p2.quantile(0.25)
    Q3 = temp_p2.quantile(0.75)
    IQR = Q3 - Q1
    seuils = []
    seuils.append(Q1 - 1.5 * IQR)
    seuils.append(Q3 + 1.5 * IQR)

    return seuils


def extreme_value_surface(df):
    """
    Calcule les seuils bas et hauts basés sur les données réelles de la surface.
    """
    Q1 = df["surface_reelle_bati"].quantile(0.25)
    Q3 = df["surface_reelle_bati"].quantile(0.75)
    IQR = Q3 - Q1
    seuils = []
    seuils.append((Q1 - 1.5 * IQR))
    seuils.append((Q3 + 1.5 * IQR))

    return seuils


def verify_dvf_columns(df_list, years_labels):
    """
    Vérifie si toutes les colonnes sont identiques entre les différents DataFrames.
    Affiche un diagnostic précis en cas de différence.
    """
    reference_cols = set(df_list[0].columns)
    all_match = True

    print("--- Diagnostic de cohérence des colonnes ---")
    for df, year in zip(df_list, years_labels):
        current_cols = set(df.columns)
        if current_cols == reference_cols:
            print(f"Année {year} : Colonnes conformes.")
        else:
            all_match = False
            missing = reference_cols - current_cols
            extra = current_cols - reference_cols
            print(f"Année {year} : Différences détectées !")
            if missing:
                print(f"   - Manquantes : {missing}")
            if extra:
                print(f"   - En trop : {extra}")

    return all_match
