import streamlit as st
import pandas as pd
import numpy as np
# Importation conditionnelle de matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    # Continuer sans matplotlib, car nous utilisons plotly pour les graphiques
    pass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Simulateur LMNP",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre et introduction de l'application
st.title("Simulateur LMNP - Faut-il rester en LMNP ?")
st.markdown("""
Cette application est un simulateur basé sur le fichier Excel 'Faut-il rester en LMNP'.  
**Remplissez uniquement les champs en bleu**, l'application s'adaptera à votre situation.
""")

# ----- FONCTIONS DE CALCUL -----
# Ces fonctions reproduisent la logique des calculs du fichier Excel

def calculer_frais_notaire(prix_achat, taux_frais_notaire):
    """Calcule les frais de notaire."""
    return prix_achat * taux_frais_notaire

def calculer_prix_acquisition(prix_achat, travaux, frais_notaire):
    """Calcule le prix d'acquisition total."""
    return prix_achat + travaux + frais_notaire

def calculer_loyer_annuel(loyer_mensuel, vacance_locative):
    """Calcule le loyer annuel en tenant compte de la vacance locative."""
    return loyer_mensuel * (12 - vacance_locative)

def calculer_charges_annuelles(taxe_fonciere, charges_copro, entretien_courant, 
                               gestion_locative, comptable, gros_entretien, 
                               assurance_pno, autres_depenses):
    """Calcule les charges annuelles totales."""
    return (taxe_fonciere + charges_copro + entretien_courant + 
            gestion_locative * 12 + comptable * 12 + gros_entretien + 
            assurance_pno + autres_depenses)

def calculer_mensualite_pret(montant_emprunte, taux_emprunt, duree_emprunt, taux_assurance):
    """Calcule la mensualité du prêt (capital + intérêts + assurance)."""
    taux_mensuel = taux_emprunt / 12
    nombre_mensualites = duree_emprunt * 12
    
    # Formule de calcul des mensualités (capital + intérêts)
    if taux_mensuel > 0:
        mensualite = montant_emprunte * taux_mensuel * (1 + taux_mensuel) ** nombre_mensualites / ((1 + taux_mensuel) ** nombre_mensualites - 1)
    else:
        mensualite = montant_emprunte / nombre_mensualites
    
    # Ajouter l'assurance emprunteur
    mensualite_assurance = montant_emprunte * taux_assurance / 12
    
    return mensualite + mensualite_assurance

def calculer_interets_premiere_annee(montant_emprunte, taux_emprunt, duree_emprunt):
    """Calcule les intérêts payés la première année."""
    taux_mensuel = taux_emprunt / 12
    nombre_mensualites = duree_emprunt * 12
    
    if taux_mensuel <= 0:
        return 0
    
    mensualite = montant_emprunte * taux_mensuel * (1 + taux_mensuel) ** nombre_mensualites / ((1 + taux_mensuel) ** nombre_mensualites - 1)
    
    # Calculer les intérêts pour la première année
    interets_premiere_annee = 0
    capital_restant = montant_emprunte
    
    for i in range(12):
        interet_mois = capital_restant * taux_mensuel
        amortissement_capital = mensualite - interet_mois
        capital_restant -= amortissement_capital
        interets_premiere_annee += interet_mois
    
    return interets_premiere_annee

def calculer_amortissements(prix_achat, travaux, mobilier, part_terrain, 
                           duree_amort_bien, duree_amort_travaux, duree_amort_mobilier):
    """Calcule les amortissements annuels."""
    # Valeur du bâti = prix d'achat - valeur du terrain
    valeur_bati = prix_achat * (1 - part_terrain)
    
    # Calcul des amortissements annuels
    amort_bati = valeur_bati / duree_amort_bien if duree_amort_bien > 0 else 0
    amort_travaux = travaux / duree_amort_travaux if duree_amort_travaux > 0 else 0
    amort_mobilier = mobilier / duree_amort_mobilier if duree_amort_mobilier > 0 else 0
    
    return {
        "bati": amort_bati,
        "travaux": amort_travaux,
        "mobilier": amort_mobilier,
        "total": amort_bati + amort_travaux + amort_mobilier
    }

def calculer_resultat_fiscal_lmnp(loyer_annuel, charges_annuelles, interets_emprunt, amortissements=None):
    """Calcule le résultat fiscal en LMNP."""
    resultat_hors_amort = loyer_annuel - charges_annuelles - interets_emprunt
    
    # Si un montant d'amortissement est fourni, on l'utilise pour le régime réel
    if amortissements and resultat_hors_amort > 0:
        # On utilise les amortissements seulement jusqu'à hauteur du résultat positif
        amort_utilises = min(amortissements["total"], resultat_hors_amort)
        resultat_fiscal = resultat_hors_amort - amort_utilises
        amort_reportables = amortissements["total"] - amort_utilises
    else:
        resultat_fiscal = resultat_hors_amort
        amort_reportables = amortissements["total"] if amortissements else 0
    
    return {
        "resultat_hors_amort": resultat_hors_amort,
        "resultat_fiscal": resultat_fiscal,
        "amort_utilises": amort_utilises if amortissements and resultat_hors_amort > 0 else 0,
        "amort_reportables": amort_reportables
    }

def calculer_economie_impots(resultat_fiscal, tmi):
    """Calcule l'économie d'impôts liée au déficit foncier."""
    if resultat_fiscal < 0:
        # Plafond du déficit foncier imputable sur le revenu global = 10 700€
        deficit_imputable = min(abs(resultat_fiscal), 10700)
        economie_impots = deficit_imputable * tmi
    else:
        economie_impots = 0
    
    return economie_impots

def calculer_impot_revenu_lmnp(resultat_fiscal, tmi):
    """Calcule l'impôt sur le revenu en régime LMNP."""
    if resultat_fiscal > 0:
        return resultat_fiscal * tmi
    else:
        return 0

def calculer_cashflow_mensuel(loyer_mensuel, charges_annuelles, mensualite_pret, economie_impots=0):
    """Calcule le cash-flow mensuel."""
    charges_mensuelles = charges_annuelles / 12
    economie_impots_mensuelle = economie_impots / 12
    
    return loyer_mensuel - charges_mensuelles - mensualite_pret + economie_impots_mensuelle

def calculer_rentabilite(cashflow_annuel, prix_acquisition):
    """Calcule la rentabilité de l'investissement."""
    if prix_acquisition > 0:
        return cashflow_annuel / prix_acquisition * 100
    else:
        return 0

def calculer_tableau_amortissement_pret(montant_emprunte, taux_annuel, duree_annees):
    """Génère le tableau d'amortissement du prêt."""
    taux_mensuel = taux_annuel / 12
    nb_mensualites = duree_annees * 12
    
    if taux_mensuel <= 0:
        mensualite = montant_emprunte / nb_mensualites
    else:
        mensualite = montant_emprunte * taux_mensuel * (1 + taux_mensuel) ** nb_mensualites / ((1 + taux_mensuel) ** nb_mensualites - 1)
    
    tableau = []
    capital_restant = montant_emprunte
    
    for i in range(1, nb_mensualites + 1):
        interet = capital_restant * taux_mensuel
        amortissement = mensualite - interet
        capital_restant -= amortissement
        
        tableau.append({
            "mois": i,
            "mensualite": mensualite,
            "amortissement": amortissement,
            "interet": interet,
            "capital_restant": max(0, capital_restant)  # Éviter les valeurs négatives dues aux arrondis
        })
    
    return pd.DataFrame(tableau)

def projeter_sur_duree(prix_achat, travaux, mobilier, loyer_mensuel, charges_annuelles, 
                     apport, duree_emprunt, taux_emprunt, taux_assurance, duree_projection, 
                     taux_augmentation_bien, taux_augmentation_loyer=0.01, taux_augmentation_charges=0.02,
                     part_terrain=0.1, duree_amort_bien=35, duree_amort_travaux=12, duree_amort_mobilier=6, tmi=0.11):
    """Projette l'investissement sur la durée spécifiée."""
    # Initialisation
    frais_notaire = calculer_frais_notaire(prix_achat, 0.075)  # 7.5% par défaut
    prix_acquisition = calculer_prix_acquisition(prix_achat, travaux, frais_notaire)
    montant_emprunte = prix_acquisition - apport
    mensualite = calculer_mensualite_pret(montant_emprunte, taux_emprunt, duree_emprunt, taux_assurance)
    loyer_annuel = calculer_loyer_annuel(loyer_mensuel, vacance_locative)
    
    # Calcul des intérêts de la première année
    interets_premiere_annee = calculer_interets_premiere_annee(montant_emprunte, taux_emprunt, duree_emprunt)
    
    # Calcul des amortissements
    amortissements = calculer_amortissements(
        prix_achat, travaux, mobilier, part_terrain, 
        duree_amort_bien, duree_amort_travaux, duree_amort_mobilier
    )
    
    # Calcul du résultat fiscal LMNP
    resultat_fiscal = calculer_resultat_fiscal_lmnp(loyer_annuel, charges_annuelles, interets_premiere_annee, amortissements)
    
    # Calcul des économies d'impôts
    economie_impots = calculer_economie_impots(resultat_fiscal["resultat_fiscal"], tmi)
    
    # Calcul du cash-flow mensuel
    cashflow_mensuel = calculer_cashflow_mensuel(loyer_mensuel, charges_annuelles, mensualite, economie_impots)
    
    # Calcul de la rentabilité
    rentabilite = calculer_rentabilite(cashflow_mensuel * 12, prix_acquisition)
    
    # Afficher les résultats clés dans l'onglet Entrées
    with tab1:
        st.markdown("---")
        st.subheader("📊 Résultats clés")
        
        # Créer une grille de 2x2 pour les métriques
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("Cash-flow mensuel", f"{cashflow_mensuel:.2f} €")
        with metric_cols[1]:
            st.metric("Rentabilité", f"{rentabilite:.2f} %")
        with metric_cols[2]:
            st.metric("Économie d'impôts", f"{economie_impots:.2f} €/an")
        with metric_cols[3]:
            st.metric("Mensualité du prêt", f"{mensualite:.2f} €/mois")
    
    # Projection sur la durée de détention
    projection = projeter_sur_duree(
        prix_achat, travaux, mobilier, loyer_mensuel, charges_annuelles,
        apport, duree_emprunt, taux_emprunt, taux_assurance, int(duree_detention),
        taux_augmentation_bien, taux_augmentation_loyer, taux_augmentation_charges,
        part_terrain, duree_amort_bien, duree_amort_travaux, duree_amort_mobilier, tmi
    )
    
    # Afficher les résultats détaillés dans l'onglet Résultats détaillés
    with tab2:
        st.subheader("Projection sur la durée de détention")
        
        # Graphique d'évolution de la valeur du bien
        st.plotly_chart(
            create_evolution_graph(
                projection, 
                ["valeur_bien"], 
                "Évolution de la valeur du bien", 
                "Valeur (€)",
                ["blue"]
            ),
            use_container_width=True
        )
        
        # Graphique d'évolution des revenus et charges
        st.plotly_chart(
            create_stacked_bar(
                projection,
                ["loyer_annuel", "charges_annuelles", "interets_emprunt", "mensualite_annuelle"],
                "Évolution des revenus et charges",
                "Montant (€)",
                ["green", "red", "orange", "darkred"]
            ),
            use_container_width=True
        )
        
        # Graphique d'évolution du cash-flow
        st.plotly_chart(
            create_evolution_graph(
                projection,
                ["cashflow_mensuel"],
                "Évolution du cash-flow mensuel",
                "Cash-flow (€/mois)",
                ["green"]
            ),
            use_container_width=True
        )
        
        # Graphique d'évolution de la rentabilité
        st.plotly_chart(
            create_evolution_graph(
                projection,
                ["rentabilite"],
                "Évolution de la rentabilité",
                "Rentabilité (%)",
                ["purple"]
            ),
            use_container_width=True
        )
        
        # Graphique de comparaison impôts vs économies
        st.plotly_chart(
            create_comparison_chart(
                projection,
                "impot_revenu_lmnp",
                "economie_impots",
                "Impôt sur le revenu LMNP",
                "Économies d'impôts",
                "Impôt vs Économies d'impôts",
                "Montant (€)"
            ),
            use_container_width=True
        )
        
        # Tableau des résultats détaillés
        st.subheader("Résultats année par année")
        st.dataframe(
            projection[[
                "annee", "valeur_bien", "loyer_mensuel", "charges_annuelles",
                "interets_emprunt", "resultat_fiscal", "impot_revenu_lmnp",
                "economie_impots", "cashflow_mensuel", "rentabilite"
            ]].style.format({
                "valeur_bien": "{:.2f} €",
                "loyer_mensuel": "{:.2f} €",
                "charges_annuelles": "{:.2f} €",
                "interets_emprunt": "{:.2f} €",
                "resultat_fiscal": "{:.2f} €",
                "impot_revenu_lmnp": "{:.2f} €",
                "economie_impots": "{:.2f} €",
                "cashflow_mensuel": "{:.2f} €",
                "rentabilite": "{:.2f} %"
            }),
            use_container_width=True
        )
    
    # Afficher les tableaux d'amortissement dans l'onglet Tableaux d'amortissement
    with tab3:
        st.subheader("Tableau d'amortissement du prêt")
        
        # Calculer le tableau d'amortissement
        tableau_amort = calculer_tableau_amortissement_pret(montant_emprunte, taux_emprunt, duree_emprunt)
        
        # Graphique d'évolution du capital et des intérêts
        capital_par_annee = tableau_amort.groupby(tableau_amort['mois'].apply(lambda x: (x-1)//12 + 1))['amortissement'].sum().reset_index()
        capital_par_annee.columns = ['annee', 'amortissement_capital']
        
        interets_par_annee = tableau_amort.groupby(tableau_amort['mois'].apply(lambda x: (x-1)//12 + 1))['interet'].sum().reset_index()
        interets_par_annee.columns = ['annee', 'interets']
        
        df_amort_annuel = pd.merge(capital_par_annee, interets_par_annee, on='annee')
        
        # Graphique d'évolution du capital et des intérêts
        st.plotly_chart(
            create_stacked_bar(
                df_amort_annuel,
                ["amortissement_capital", "interets"],
                "Répartition des mensualités par année",
                "Montant (€)",
                ["blue", "red"]
            ),
            use_container_width=True
        )
        
        # Tableau d'amortissement (version résumée par année)
        st.subheader("Résumé annuel")
        st.dataframe(
            df_amort_annuel.style.format({
                "amortissement_capital": "{:.2f} €",
                "interets": "{:.2f} €"
            }),
            use_container_width=True
        )
        
        # Option pour afficher le tableau complet
        if st.checkbox("Afficher le tableau d'amortissement complet (mensuel)"):
            st.subheader("Tableau d'amortissement mensuel")
            tableau_amort['mois_annee'] = tableau_amort['mois'].apply(lambda x: f"Année {(x-1)//12 + 1}, Mois {((x-1)%12) + 1}")
            st.dataframe(
                tableau_amort[['mois_annee', 'mensualite', 'amortissement', 'interet', 'capital_restant']].style.format({
                    "mensualite": "{:.2f} €",
                    "amortissement": "{:.2f} €",
                    "interet": "{:.2f} €",
                    "capital_restant": "{:.2f} €"
                }),
                use_container_width=True
            )
        
        # Tableau d'amortissement du bien
        st.subheader("Amortissements")
        
        # Résumé des amortissements
        st.markdown(f"**Bâti:** {amortissements['bati']:.2f} €/an sur {duree_amort_bien} ans")
        st.markdown(f"**Travaux:** {amortissements['travaux']:.2f} €/an sur {duree_amort_travaux} ans")
        st.markdown(f"**Mobilier:** {amortissements['mobilier']:.2f} €/an sur {duree_amort_mobilier} ans")
        st.markdown(f"**Total des amortissements annuels:** {amortissements['total']:.2f} €/an")
        
        # Graphique d'évolution des amortissements utilisés et reportés
        amort_df = projection[['annee', 'amort_utilises', 'amort_reportables']]
        
        st.plotly_chart(
            create_stacked_bar(
                amort_df,
                ["amort_utilises", "amort_reportables"],
                "Amortissements utilisés et reportables par année",
                "Montant (€)",
                ["green", "orange"]
            ),
            use_container_width=True
        )


if __name__ == "__main__":
    main(montant_emprunte, taux_emprunt, duree_emprunt, taux_assurance)
    
    projection = []
    loyer = loyer_mensuel
    charges = charges_annuelles
    valeur_bien = prix_achat + travaux
    
    # Calcul du tableau d'amortissement pour obtenir les intérêts année par année
    tableau_amort = calculer_tableau_amortissement_pret(montant_emprunte, taux_emprunt, duree_emprunt)
    
    # Calcul des amortissements annuels (constants sur la durée)
    amortissements_annuels = calculer_amortissements(
        prix_achat, travaux, mobilier, part_terrain, 
        duree_amort_bien, duree_amort_travaux, duree_amort_mobilier
    )
    
    # Variables pour le suivi des amortissements reportés
    amort_reportes_cumules = 0
    
    for annee in range(1, duree_projection + 1):
        # Mise à jour des valeurs avec inflation
        loyer = loyer_mensuel * (1 + taux_augmentation_loyer) ** (annee - 1)
        charges = charges_annuelles * (1 + taux_augmentation_charges) ** (annee - 1)
        valeur_bien = (prix_achat + travaux) * (1 + taux_augmentation_bien) ** (annee - 1)
        
        # Calcul des intérêts payés cette année
        debut_mois = (annee - 1) * 12 + 1
        fin_mois = annee * 12
        if debut_mois <= len(tableau_amort):
            interets_annee = tableau_amort.loc[(tableau_amort['mois'] >= debut_mois) & 
                                            (tableau_amort['mois'] <= fin_mois), 'interet'].sum()
        else:
            interets_annee = 0
        
        loyer_annuel = loyer * 12
        
        # Calcul du résultat fiscal LMNP
        amort_disponibles = {
            "bati": amortissements_annuels["bati"],
            "travaux": amortissements_annuels["travaux"],
            "mobilier": amortissements_annuels["mobilier"],
            "total": amortissements_annuels["total"] + amort_reportes_cumules
        }
        
        resultat = calculer_resultat_fiscal_lmnp(loyer_annuel, charges, interets_annee, amort_disponibles)
        
        # Mise à jour des amortissements reportés
        amort_reportes_cumules = resultat["amort_reportables"]
        
        # Calcul des économies d'impôts
        economie_impots = calculer_economie_impots(resultat["resultat_fiscal"], tmi)
        
        # Calcul de l'impôt sur le revenu LMNP
        impot_revenu_lmnp = calculer_impot_revenu_lmnp(resultat["resultat_fiscal"], tmi)
        
        # Calcul du cash-flow
        cashflow_mensuel = calculer_cashflow_mensuel(loyer, charges, mensualite if annee <= duree_emprunt else 0, economie_impots)
        cashflow_annuel = cashflow_mensuel * 12
        
        # Ajouter les données de l'année à la projection
        projection.append({
            "annee": annee,
            "valeur_bien": valeur_bien,
            "loyer_mensuel": loyer,
            "loyer_annuel": loyer_annuel,
            "charges_annuelles": charges,
            "interets_emprunt": interets_annee,
            "mensualite_annuelle": mensualite * 12 if annee <= duree_emprunt else 0,
            "resultat_hors_amort": resultat["resultat_hors_amort"],
            "amort_utilises": resultat["amort_utilises"],
            "amort_reportables": resultat["amort_reportables"],
            "resultat_fiscal": resultat["resultat_fiscal"],
            "economie_impots": economie_impots,
            "impot_revenu_lmnp": impot_revenu_lmnp,
            "cashflow_mensuel": cashflow_mensuel,
            "cashflow_annuel": cashflow_annuel,
            "rentabilite": calculer_rentabilite(cashflow_annuel, prix_acquisition)
        })
    
return pd.DataFrame(projection)

# Fonction pour créer les graphiques
def create_evolution_graph(df, y_columns, title, y_axis_title, colors=None):
    """Crée un graphique d'évolution sur plusieurs années."""
    fig = go.Figure()
    
    for i, column in enumerate(y_columns):
        color = colors[i] if colors and i < len(colors) else None
        fig.add_trace(go.Scatter(
            x=df["annee"],
            y=df[column],
            mode="lines+markers",
            name=column,
            line=dict(color=color) if color else None
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Année",
        yaxis_title=y_axis_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    return fig

def create_stacked_bar(df, categories, title, y_axis_title, colors=None):
    """Crée un graphique en barres empilées."""
    fig = go.Figure()
    
    for i, category in enumerate(categories):
        color = colors[i] if colors and i < len(colors) else None
        fig.add_trace(go.Bar(
            x=df["annee"],
            y=df[category],
            name=category,
            marker_color=color
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Année",
        yaxis_title=y_axis_title,
        barmode="stack",
        height=400
    )
    
    return fig

def create_comparison_chart(df, metric1, metric2, name1, name2, title, y_axis_title):
    """Crée un graphique de comparaison entre deux métriques."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["annee"],
        y=df[metric1],
        name=name1,
        marker_color="blue"
    ))
    
    fig.add_trace(go.Bar(
        x=df["annee"],
        y=df[metric2],
        name=name2,
        marker_color="green"
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Année",
        yaxis_title=y_axis_title,
        barmode="group",
        height=400
    )
    
    return fig

def make_blue_input(label, key, value, min_value=None, max_value=None, step=None, format=None, help=None, unit=""):
    """Crée un champ de saisie en bleu pour indiquer qu'il est modifiable."""
    col1, col2 = st.columns([3, 1])
    with col1:
        result = st.number_input(
            label, 
            value=value, 
            min_value=min_value, 
            max_value=max_value,
            step=step,
            format=format,
            help=help,
            key=key
        )
    with col2:
        st.markdown(f"<p style='margin-top:30px'>{unit}</p>", unsafe_allow_html=True)
    return result

# Interface utilisateur principale avec les entrées et résultats
def main():
    # Organisation de l'interface en onglets
    tab1, tab2, tab3 = st.tabs(["Entrées", "Résultats détaillés", "Tableaux d'amortissement"])
    
    with tab1:
        # Diviser la page en deux colonnes principales
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Section Acquisition
            st.subheader("Coût d'acquisition")
            
            prix_achat = make_blue_input("Prix d'achat", "prix_achat", 250000.0, 0.0, None, 1000.0, "%.2f", "Prix d'achat du bien immobilier", "€")
            travaux = make_blue_input("Travaux", "travaux", 35000.0, 0.0, None, 1000.0, "%.2f", "Montant des travaux à réaliser", "€")
            mobilier = make_blue_input("Mobilier", "mobilier", 0.0, 0.0, None, 1000.0, "%.2f", "Montant du mobilier", "€")
            taux_frais_notaire = make_blue_input("Frais de notaire", "taux_frais_notaire", 0.075, 0.0, 0.15, 0.001, "%.3f", "Taux des frais de notaire", "%")
            
            # Calculs intermédiaires pour l'affichage
            frais_notaire = calculer_frais_notaire(prix_achat, taux_frais_notaire)
            prix_acquisition = calculer_prix_acquisition(prix_achat, travaux, frais_notaire)
            
            # Affichage des résultats calculés (non modifiables)
            st.markdown(f"**Frais de notaire:** {frais_notaire:.2f} €")
            st.markdown(f"**Prix d'acquisition total:** {prix_acquisition:.2f} €")
            
            # Section Exploitation
            st.subheader("Exploitation du bien")
            loyer_mensuel = make_blue_input("Loyer charges comprises", "loyer_mensuel", 1000.0, 0.0, None, 50.0, "%.2f", "Loyer mensuel charges comprises", "€/mois")
            vacance_locative = make_blue_input("Vacance locative", "vacance_locative", 0.0, 0.0, 12.0, 0.5, "%.1f", "Nombre de mois de vacance locative par an", "mois/an")
            
            # Section Détention
            st.subheader("Détention du bien")
            annee_possession = make_blue_input("Depuis combien de temps possédez-vous le bien ?", "annee_possession", 0.0, 0.0, None, 1.0, "%.0f", "Nombre d'années de possession du bien", "an")
            duree_detention = make_blue_input("Durée estimée de détention restante", "duree_detention", 20.0, 1.0, 50.0, 1.0, "%.0f", "Durée estimée de détention restante", "an")
            
            # Section Coûts de détention
            st.subheader("Coûts de détention")
            taxe_fonciere = make_blue_input("Taxe foncière", "taxe_fonciere", 200.0, 0.0, None, 50.0, "%.2f", "Montant annuel de la taxe foncière", "€/an")
            charges_copro = make_blue_input("Charges de copropriété", "charges_copro", 700.0, 0.0, None, 50.0, "%.2f", "Montant annuel des charges de copropriété", "€/an")
            entretien_courant = make_blue_input("Entretien courant", "entretien_courant", 500.0, 0.0, None, 50.0, "%.2f", "Montant annuel pour l'entretien courant", "€/an")
            gestion_locative = make_blue_input("Gestion locative", "gestion_locative", 0.0, 0.0, None, 10.0, "%.2f", "Montant mensuel de la gestion locative", "€/mois")
            comptable = make_blue_input("Comptable", "comptable", 50.0, 0.0, None, 10.0, "%.2f", "Montant mensuel des frais de comptable", "€/mois")
            gros_entretien = make_blue_input("Gros entretien", "gros_entretien", 0.0, 0.0, None, 100.0, "%.2f", "Provision annuelle pour gros entretien", "€/an")
            assurance_pno = make_blue_input("Assurance PNO", "assurance_pno", 250.0, 0.0, None, 50.0, "%.2f", "Montant annuel de l'assurance PNO", "€/an")
            autres_depenses = make_blue_input("Autres dépenses", "autres_depenses", 0.0, 0.0, None, 50.0, "%.2f", "Montant annuel des autres dépenses", "€/an")
            
            # Section Financement
            st.subheader("Financement")
            apport = make_blue_input("Apport", "apport", 10000.0, 0.0, None, 1000.0, "%.2f", "Montant de l'apport", "€")
            
            # Calcul du montant emprunté
            montant_emprunte = prix_acquisition - apport
            st.markdown(f"**Montant emprunté:** {montant_emprunte:.2f} €")
            
            duree_emprunt = make_blue_input("Durée d'emprunt", "duree_emprunt", 15.0, 1.0, 30.0, 1.0, "%.0f", "Durée de l'emprunt", "an")
            taux_emprunt = make_blue_input("Taux d'emprunt", "taux_emprunt", 0.03, 0.0, 0.1, 0.001, "%.3f", "Taux d'intérêt annuel de l'emprunt", "%")
            taux_assurance = make_blue_input("Taux d'assurance emprunteur", "taux_assurance", 0.0035, 0.0, 0.01, 0.0001, "%.4f", "Taux d'assurance emprunteur annuel", "%")
            
        with col_right:
            # Section Imposition
            st.subheader("Imposition")
            revenus_annuels = make_blue_input("Revenus annuels", "revenus_annuels", 25000.0, 0.0, None, 1000.0, "%.2f", "Revenus annuels du foyer fiscal", "€/an")
            
            # Options pour situation familiale
            conjoint = st.selectbox("Conjoint", ["Non", "Oui"], index=0)
            enfants = st.selectbox("Enfants", ["Non", "Oui"], index=0)
            marie_pacse = st.selectbox("Marié ou Pacsé", ["Non", "Oui"], index=0)
            
            # Nombre de parts fiscales
            nombre_parts = 1
            if marie_pacse == "Oui":
                nombre_parts += 1
            
            nb_enfants = make_blue_input("Nombre d'enfants", "nb_enfants", 0, 0, 10, 1, "%d", "Nombre d'enfants à charge", "")
            
            # Calcul des parts fiscales
            parts_enfants = 0
            if nb_enfants > 0:
                for i in range(1, nb_enfants + 1):
                    if i <= 2:
                        parts_enfants += 0.5
                    else:
                        parts_enfants += 1
            
            nombre_parts += parts_enfants
            st.markdown(f"**Nombre de parts fiscales:** {nombre_parts}")
            
            # TMI (Taux marginal d'imposition)
            tmi = make_blue_input("TMI", "tmi", 0.11, 0.0, 0.45, 0.01, "%.2f", "Taux marginal d'imposition", "")
            
            # Section Hypothèses amortissement
            st.subheader("Hypothèses amortissement")
            part_terrain = make_blue_input("Part du terrain dans la valeur du bien", "part_terrain", 0.1, 0.0, 1.0, 0.01, "%.2f", "Part du terrain dans la valeur du bien", "%")
            duree_amort_bien = make_blue_input("Durée amortissement du bien", "duree_amort_bien", 35, 1, 50, 1, "%d", "Durée d'amortissement du bâti (années)", "an")
            duree_amort_travaux = make_blue_input("Durée d'amortissement des travaux", "duree_amort_travaux", 12, 1, 30, 1, "%d", "Durée d'amortissement des travaux (années)", "an")
            duree_amort_mobilier = make_blue_input("Durée d'amortissement du mobilier", "duree_amort_mobilier", 6, 1, 10, 1, "%d", "Durée d'amortissement du mobilier (années)", "an")
            
            # Section Environnement de marché
            st.subheader("Environnement de marché")
            taux_augmentation_bien = make_blue_input("Augmentation prix de votre bien", "taux_augmentation_bien", 0.02, -0.1, 0.1, 0.005, "%.3f", "Taux annuel d'augmentation de la valeur du bien", "%")
            taux_augmentation_loyer = make_blue_input("Augmentation des loyers", "taux_augmentation_loyer", 0.01, -0.05, 0.1, 0.005, "%.3f", "Taux annuel d'augmentation des loyers", "%")
            taux_augmentation_charges = make_blue_input("Augmentation des charges", "taux_augmentation_charges", 0.02, -0.05, 0.1, 0.005, "%.3f", "Taux annuel d'augmentation des charges", "%")
            
    # Calculs principaux
    charges_annuelles = calculer_charges_annuelles(
        taxe_fonciere, charges_copro, entretien_courant,
        gestion_locative, comptable, gros_entretien,
        assurance_pno, autres_depenses
    )
    
    mensualite = calculer_mens
