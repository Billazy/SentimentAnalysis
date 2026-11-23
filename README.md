# Sentiment Analysis with ML.NET

Classification binaire (positif/négatif) d'avis produits avec ML.NET.

## Dataset

Deux datasets au format TSV (`phrase<TAB>label 0/1`) :

| Dataset | Fichier | Taille | Domaine |
|---------|---------|--------|---------|
| Yelp | `Data/yelp_labelled.txt` | ~1000 lignes | Avis restaurants |
| Amazon | `Data/amazon_cells_labelled.txt` | ~1000 lignes | Avis produits électroniques |

## Pipeline

1. `FeaturizeText` : texte → vecteur numérique (TF-IDF/hashing)
   - Unigrammes + bigrammes de mots (`NgramLength = 2, UseAllLengths = true`)
   - Optionnel : 3-grammes de caractères (config A)
2. `SdcaLogisticRegression` : classification binaire

## Résultats

### Validation croisée 10-fold (seed=42)

| Dataset | Config | Accuracy | AUC | F1 |
|---------|--------|:--------:|:---:|:--:|
| Amazon | A (bigrammes + char-ngrams) | 79,50% ± 2,64% | 90,02% | 79,23% |
| Amazon | **B (bigrammes seuls)** | **79,69% ± 2,61%** | **90,03%** | **79,49%** |

**Test t apparié** (A vs B) : t = −0,661, p ≈ 0,500 → **pas de différence significative**.

**Conclusion** : les char-ngrams n'apportent pas de gain mesurable sur Amazon.
Config B retenue pour sa simplicité.

### Comparaison Yelp vs Amazon

| Dataset | Meilleure accuracy CV | Difficulté |
|---------|:---------------------:|------------|
| Yelp | ~83,96% | Plus facile (avis bien écrits) |
| Amazon | ~79,69% | Plus difficile (bruit d'annotation, fautes de frappe) |

**Note** : Amazon contient ~5% d'étiquettes douteuses (ex. *"Excellent starter
wireless headset."* étiqueté `0`), ce qui plafonne l'accuracy atteignable.

### Analyse des seuils (Amazon, split 80/20)

| Seuil | Accuracy |
|:-----:|:--------:|
| 0,3 | 75,94% |
| 0,4 | **78,07%** |
| 0,5 | 77,01% |
| 0,6 | 77,54% |
| 0,7 | 74,87% |

Le seuil 0,5 est conservé (standard). Le seuil optimal empirique est ~0,4–0,6.

## Méthodologie

- **Seed fixe** (`42`) pour la reproductibilité.
- **Validation croisée 10-fold** pour des mesures fiables.
- **Test t apparié** pour tester la significativité des différences.
- **Décision automatique** de la config (A ou B) selon les résultats.
- **Split 80/20** pour la mesure de production (moins fiable que la CV).

## Structure du code
