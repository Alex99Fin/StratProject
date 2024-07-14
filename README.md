# Analyse en Composantes Principales des Variations de Spreads de Crédit

    Ce projet vise à réaliser une Analyse en Composantes Principales (ACP) des variations de spreads de crédit à partir d'un ensemble de données historiques. Nous identifions les principaux facteurs explicatifs influençant ces variations.

## Données Utilisées:

    Les données pour cette étude proviennent exclusivement des indices de Bank of America Investment Grade pour les zones Euro et US. Les caractéristiques des données sont les suivantes :

        Indices : Segmentés par notation et maturité (ex. IG_AAA_Euro_1-3Y).
        Fréquence : Hebdomadaire.
        Période : Du 07 janvier 2000 au 31 mai 2024.

## Structure de l'Étude:

    L'étude est divisée en deux grandes parties géographiques (EU et US), et chaque partie est subdivisée en deux analyses :

    Analyse par Courbe de Spread selon la Maturité :

        Indices combinés de différentes notations pour chaque catégorie de maturité (ex. "1-3Y").

    Analyse par Notation :

        Indices combinés pour différentes maturités sous une même notation.


## Méthodologie

    Analyse en Composantes Principales (ACP)
    
    Objectif : 
        - Effectuer une ACP sur l'ensemble des données collectées pour extraire les principales composantes.
    
    Procédure :
        - Prétraitement des données pour normaliser les indices.
        - Application de l'ACP pour déterminer les vecteurs propres et les valeurs propres.
        - Sélection et interprétation des trois premières composantes principales.

## Régression des Composantes Principales

    Objectif :
        - Régresser les trois composantes principales sur les marchés respectifs pour analyser leur capacité à expliquer les dynamiques des périodes financières clés.

    Méthode :

        - Utilisation de techniques de régression linéaire.
        - Évaluation de la significativité des coefficients de régression.
        - Interprétation des résultats en fonction des contextes économiques et financiers pertinents.
