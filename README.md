# Projet ML — Performances des Étudiants

Projet académique réalisé dans le cadre du cours de Machine Learning (Pr. Kalloubi — UCA Marrakech).  
L'objectif est d'appliquer l'ensemble des algorithmes ML vus en cours sur un dataset réel de performances scolaires.

---

## Dataset

**StudentsPerformance.csv** — 1 000 étudiants, 8 variables :

| Variable | Type | Description |
|---|---|---|
| `gender` | Catégorielle | Genre de l'étudiant |
| `race/ethnicity` | Catégorielle | Groupe ethnique (A–E) |
| `parental level of education` | Ordinale | Niveau d'études des parents |
| `lunch` | Catégorielle | Type de repas (standard / free-reduced) |
| `test preparation course` | Catégorielle | Préparation à l'examen (completed / none) |
| `math score` | Numérique | Score en mathématiques (0–100) |
| `reading score` | Numérique | Score en lecture (0–100) |
| `writing score` | Numérique | Score en rédaction (0–100) |

---

## Structure du notebook

```
ML_StudentsPerformance.ipynb
│
├── 0. Imports
├── 1. Exploration & Préparation (EDA)
│   ├── Chargement et aperçu
│   ├── Visualisations exploratoires
│   └── Feature engineering & encodage
│
├── 2. Modèles Linéaires
│   ├── Régression : OLS, Ridge, Lasso, Elastic-Net
│   └── Classification : Régression Logistique (Pass/Fail)
│
├── 3. Arbres de Décision
│   ├── CART (Gini) — classification & régression
│   └── Analyse biais-variance (max_depth)
│
├── 4. K-Nearest Neighbors (KNN)
│   ├── Sélection du K optimal par cross-validation
│   └── Comparaison des métriques de distance
│
├── 5. Ensemble Learning
│   ├── Random Forest (Bagging)
│   ├── AdaBoost (Boosting)
│   └── Gradient Boosting
│
├── 6. Méthodes Rule-Based
│   ├── Extraction de règles depuis arbre de décision
│   ├── Calcul Coverage / Accuracy des règles
│   └── RIPPER (si wittgenstein disponible)
│
├── 7. Clustering (Non Supervisé)
│   ├── K-Means + Méthode du Coude
│   ├── Clustering Hiérarchique + Dendrogramme
│   └── DBSCAN
│
└── 8. Évaluation Complète
    ├── Cross-Validation StratifiedKFold (5-fold)
    ├── Courbes ROC / AUC
    └── Tableau comparatif des modèles
```

---

## Résultats principaux

### Classification (Pass/Fail — seuil avg ≥ 60)

| Modèle | Accuracy | F1-Score (CV 5-fold) |
|---|---|---|
| Gradient Boosting | ~0.82 | meilleur |
| Random Forest | ~0.81 | — |
| AdaBoost | ~0.80 | — |
| Logistic Regression | ~0.79 | — |
| Decision Tree | ~0.77 | — |
| KNN | ~0.76 | — |

### Régression (Math Score)

| Modèle | R² |
|---|---|
| Random Forest | meilleur |
| Gradient Boosting | — |
| Ridge / Lasso / Elastic-Net | comparables |

### Clustering (K-Means, K=3)

3 profils naturels d'étudiants identifiés : **faible / moyen / fort**  
Silhouette Score ≈ 0.45–0.55

---

## Insights clés

- **Le type de repas (`lunch`)** est la variable la plus prédictive des performances
- **La préparation à l'examen** améliore les scores d'environ **+6 points** en moyenne
- **Reading et Writing** sont très fortement corrélés (r = 0.955)
- **Les modèles d'ensemble** (Random Forest, Gradient Boosting) surpassent systématiquement les modèles simples
- **K-Means (K=3)** révèle 3 groupes naturels cohérents avec les niveaux académiques

---

## Lancer le projet

### Prérequis

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
pip install wittgenstein  # optionnel — pour RIPPER
```

### Exécution

```bash
# Cloner le repo
git clone https://github.com/Farnane2005/ML-Students-Performance.git
cd ML-Students-Performance

# Lancer Jupyter
jupyter notebook ML_StudentsPerformance.ipynb
```

> Assurez-vous que `StudentsPerformance.csv` se trouve dans le même répertoire que le notebook.

---

## Technologies utilisées

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow)

- **Python 3.10+**
- **scikit-learn** — modèles ML, métriques, pipelines
- **pandas / numpy** — manipulation de données
- **matplotlib / seaborn** — visualisations
- **scipy** — clustering hiérarchique
- **wittgenstein** *(optionnel)* — algorithme RIPPER

---

## Auteur

Projet réalisé par **EL FARNANE MOHAMED**  
Cours : Machine Learning — Pr. Kalloubi
