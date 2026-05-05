# Assignment 1 — Définition du projet

## Description du projet

Ce projet consiste à construire un modèle de **Expected Goals (xG)** appliqué au football. L'objectif est de prédire, à partir des caractéristiques d'un tir, la probabilité qu'il se transforme en but.

Le modèle xG est aujourd'hui un standard dans l'analyse football professionnelle : il permet d'évaluer la qualité réelle des occasions créées, indépendamment du score final qui peut être trompeur sur un petit échantillon de matchs.

---

## Objectif business

Dans le football moderne, le score final ne reflète pas toujours la performance réelle d'une équipe. Un xG model permet de :

- **Évaluer la qualité des occasions** : une équipe peut gagner 1-0 en ayant eu moins de bonnes occasions que l'adversaire
- **Mesurer la performance d'un attaquant** : comparer ses buts réels à son xG attendu pour détecter s'il surperforme ou sous-performe
- **Optimiser les stratégies offensives** : identifier les zones du terrain et les types d'actions qui génèrent les meilleures chances de but
- **Aide à la décision pour le recrutement** : un joueur qui marque moins que son xG est potentiellement sous-coté

---

## Définition du problème

| Élément | Valeur |
|---|---|
| Type de problème | **Classification binaire** |
| Variable cible | `is_goal` — 1 si le tir devient un but, 0 sinon |
| Niveau d'analyse | Un tir = une observation |

Il s'agit d'une classification binaire : pour chaque tir, on prédit une probabilité entre 0 et 1 représentant la chance que ce tir devienne un but.

---

## Contexte machine learning

Le modèle prend en entrée les caractéristiques d'un tir et produit une **probabilité de but**. On compare plusieurs modèles allant du plus simple au plus complexe :

1. **Logistic Regression** — baseline interprétable
2. **Random Forest** — ensemble de décision, capture les non-linéarités
3. **Gradient Boosting** — meilleure performance attendue sur ce type de données tabulaires

Le challenge principal est le **class imbalance** : seulement environ 10% des tirs deviennent des buts. Cela impose des précautions sur le split et le choix des métriques.

---

## Dataset choisi

**Nom** : StatsBomb Open Data

**Source** : [github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)

**Accès** : données gratuites et publiques, accessibles via la librairie Python `statsbombpy` sans authentification

**Type** : données événementielles (event data) — chaque action d'un match est enregistrée avec ses coordonnées spatiales précises sur un terrain de 120×80 unités

**Compétitions chargées** :
- La Liga 2015/16 (`competition_id=11, season_id=27`)
- Champions League 2018/19 (`competition_id=16, season_id=4`)
- FIFA World Cup 2018 (`competition_id=43, season_id=3`)

**Volume** : plusieurs dizaines de milliers de tirs après filtrage

**Comment obtenir les données** :
```bash
pip install statsbombpy
```
```python
from statsbombpy import sb
events = sb.competition_events(competition_id=11, season_id=27)
shots = events[events['type'] == 'Shot']
```

**Localisation dans le repository** :
- Données brutes générées : `data/raw/shots_raw.csv` (non pushé sur GitHub, voir `.gitignore`)
- Le notebook `notebooks/data_exploration.ipynb` génère et sauvegarde ce fichier automatiquement

---

## Description des features disponibles

### Features brutes (StatsBomb)

| Feature | Description | Type |
|---|---|---|
| `location` | Coordonnées XY du tir sur le terrain | Liste [x, y] |
| `shot_outcome` | Résultat du tir (Goal, Saved, Off T, Blocked) | Catégorielle |
| `shot_technique` | Technique utilisée (Normal, Head, Volley...) | Catégorielle |
| `shot_type` | Contexte du tir (Open Play, Penalty, Free Kick...) | Catégorielle |
| `under_pressure` | Pression défensive au moment du tir | Booléen |
| `play_pattern` | Pattern de jeu ayant mené au tir | Catégorielle |

### Features dérivées (calculées manuellement)

| Feature | Formule | Justification |
|---|---|---|
| `distance_to_goal` | √((x−120)² + (y−40)²) | Plus on est proche du but, plus la probabilité augmente |
| `angle_to_goal` | \|arctan2(44−y, x−120) − arctan2(36−y, x−120)\| × 180/π | Un angle large = tir face au but |
| `is_header` | 1 si `shot_technique == "Head"` | Les têtes ont un taux de conversion plus faible |
| `is_penalty` | 1 si `shot_type == "Penalty"` | ~75% de conversion, signal très fort |
| `is_free_kick` | 1 si `shot_type == "Free Kick"` | Contexte différent du jeu ouvert |
| `is_open_play` | 1 si `shot_type == "Open Play"` | Tirs en situation de jeu normale |

---

## Premières analyses exploratoires (EDA)

Le notebook `notebooks/data_exploration.ipynb` contient l'ensemble de l'EDA. Principaux résultats :

**Taux de conversion global** : environ 10% des tirs deviennent des buts → dataset fortement déséquilibré

**Distance** : les buts sont marqués en moyenne bien plus près du but que les tirs non convertis. La grande majorité des buts vient de moins de 20 unités.

**Angle** : les buts ont en moyenne un angle plus large (tir plus face au but) que les tirs non convertis.

**Technique** : les pénaltys convertissent à ~75%, les têtes et les tirs normaux en jeu ouvert ont des taux bien inférieurs.

**Pression défensive** : les tirs sous pression convertissent moins bien que les tirs sans pression, ce qui valide l'intuition.

**Localisation spatiale** : la grande majorité des buts provient de la surface de réparation, avec une concentration autour du point de pénalty et du centre de la surface.

---

## Métrique et fonction de coût envisagées

On n'utilise **pas l'accuracy** comme métrique principale — avec 10% de buts, un modèle qui prédit toujours "pas but" atteindrait 90% d'accuracy sans aucune utilité.

| Métrique | Justification |
|---|---|
| **AUC-ROC** | Mesure la capacité du modèle à distinguer buts et non-buts, indépendamment du seuil. Métrique standard pour les problèmes déséquilibrés. |
| **Log-Loss** | Pénalise les probabilités mal calibrées — essentiel pour un modèle xG dont l'output est une probabilité. |
| **Brier Score** | Erreur quadratique moyenne sur les probabilités prédites. Complémentaire au Log-Loss. |

---

## Hypothèses, risques et limites identifiées

**Hypothèses :**
- La probabilité de but dépend principalement de la position et du contexte immédiat du tir
- Les données StatsBomb sont collectées de manière cohérente entre les compétitions

**Risques :**
- **Class imbalance** : ~10% de buts seulement → nécessite un `stratify` dans le train/test split et des métriques adaptées
- **Biais de compétition** : les données proviennent de 3 compétitions spécifiques, le modèle pourrait mal généraliser à d'autres ligues ou époques
- **Features manquantes** : StatsBomb ne renseigne pas toujours tous les champs pour tous les matchs (données plus anciennes moins complètes)
- **Absence de contexte tactique** : le modèle ne connaît pas la position des défenseurs ni du gardien, contrairement aux modèles xG professionnels qui utilisent les freeze frames

**Limites :**
- Le modèle prédit une probabilité mais ne capture pas les facteurs humains (fatigue, pression psychologique, qualité du gardien adverse)
- Les pénaltys sont très faciles à prédire (~75%) et pourraient artificiellement gonfler les performances

---

## Données et notebooks

| Fichier | Rôle | Comment l'utiliser |
|---|---|---|
| `notebooks/data_exploration.ipynb` | EDA complète, génération du CSV brut | `jupyter notebook notebooks/data_exploration.ipynb` |
| `data/raw/shots_raw.csv` | Dataset brut des tirs (non pushé) | Généré automatiquement par le notebook ci-dessus |
| `src/data.py` | Pipeline de chargement et feature engineering | Importé par `scripts/main.py` |

**Pour reproduire l'EDA depuis zéro :**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/data_exploration.ipynb
# Exécuter toutes les cellules (Kernel > Restart & Run All)
```

