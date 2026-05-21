# 🏔️ Intel Image Classification — Deep Learning

> Classification automatique de paysages naturels par réseaux de neurones convolutifs  
> PyTorch · PyTorch Lightning · Google Colab · Transfer Learning

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Le problème](#le-problème)
3. [Jeu de données](#jeu-de-données)
4. [Pipeline de données](#pipeline-de-données)
5. [Architectures](#architectures)
6. [Entraînement](#entraînement)
7. [Performances](#performances)
8. [Structure du projet](#structure-du-projet)
9. [Comment exécuter](#comment-exécuter)
10. [Dépendances](#dépendances)

---

## Vue d'ensemble

Ce projet implémente une solution de **classification d'images multi-classes** sur le jeu de données Intel Image Classification. Deux architectures sont comparées :

- `ImprovedCNN` — un réseau convolutif entraîné **from scratch** (4 blocs, BatchNorm)
- `ResNet18` — un modèle pré-entraîné sur ImageNet soumis à un **fine-tuning complet** avec learning rates différentiels

Le notebook final (`Image_Classification_FINAL.ipynb`) est la version aboutie d'un processus itératif documenté en 3 drafts progressifs, chacun corrigeant un défaut précis de la version précédente.

---

## Le problème

**Tâche** : Classification supervisée en **6 catégories** de paysages à partir d'images RGB.

**Difficulté** : Certaines classes sont visuellement proches (`glacier` / `mountain`, `buildings` / `street`), ce qui exige une architecture capable de capturer des caractéristiques de haut niveau et des textures fines.

**Approche** : Comparaison d'un CNN entraîné from scratch vs. un modèle pré-entraîné (ResNet18) pour quantifier le gain du transfer learning sur ce domaine.

---

## Jeu de données

| Paramètre | Valeur |
|---|---|
| Source | [Intel Image Classification — Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) |
| Taille totale | 17 034 images |
| Jeu d'entraînement | 14 034 images |
| Jeu de test | 3 000 images |
| Résolution originale | ~150 × 150 px (variable) |
| Format | JPEG / PNG (RGB) |
| Classes | 6 |

### Classes

| Label | Description |
|---|---|
| `buildings` | Bâtiments urbains |
| `forest` | Forêts |
| `glacier` | Glaciers |
| `mountain` | Montagnes |
| `sea` | Mers et océans |
| `street` | Rues et routes |

Le jeu de données est **approximativement équilibré** entre les 6 classes (~2 300 images par classe en entraînement).

---

## Pipeline de données

### Split train / validation / test

Le jeu d'entraînement fourni est découpé en **80 % train / 20 % validation** de façon reproductible (`seed=42`). Le jeu de test reste entièrement séparé et n'est utilisé qu'à l'évaluation finale.

```
14 034 images  →  train : 11 228  |  val : 2 806
 3 000 images  →  test  :  3 000  (jamais vu pendant l'entraînement)
```

### Augmentation de données (train uniquement)

```python
transforms.Compose([
    transforms.Resize((170, 170)),
    transforms.RandomResizedCrop(150, scale=(0.8, 1.0)),   # zoom variable
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])        # normalisation ImageNet
])
```

**Validation & Test** — resize + normalisation uniquement (sans augmentation aléatoire).

---

## Architectures

### Modèle 1 — `ImprovedCNN` (from scratch)

CNN entraîné entièrement from scratch. Conçu pour des images 150 × 150 px.

```
Input (3 × 150 × 150)
  ↓
Bloc 1  — Conv2d(3→32)   + BatchNorm + ReLU + MaxPool   →  32 × 75 × 75
Bloc 2  — Conv2d(32→64)  + BatchNorm + ReLU + MaxPool   →  64 × 37 × 37
Bloc 3  — Conv2d(64→128) + BatchNorm + ReLU + MaxPool   → 128 × 18 × 18
Bloc 4  — Conv2d(128→256)+ BatchNorm + ReLU + MaxPool   → 256 ×  9 ×  9
  ↓
Flatten → Linear(256×9×9 → 256) → ReLU → Dropout(0.4)
        → Linear(256 → 128)     → ReLU → Dropout(0.3)
        → Linear(128 → 6)
```

**Paramètres totaux : 423 046**

### Modèle 2 — `ResNet18` (fine-tuning complet)

Backbone ResNet18 pré-entraîné sur ImageNet. La couche `fc` finale est remplacée par une tête de classification adaptée à 6 classes. **Le backbone est entièrement dégelé** et soumis à un learning rate différentiel.

```
ResNet18 backbone (pré-entraîné ImageNet, dégelé)
  ↓
AdaptiveAvgPool2d → Flatten
  ↓
Linear(512 → 256) → ReLU → Dropout(0.3)
Linear(256 → 6)
```

**Paramètres totaux : 11 309 382**  
**Learning rates** : backbone `1e-4` · tête de classification `1e-3`

---

## Entraînement

| Hyperparamètre | `ImprovedCNN` | `ResNet18` |
|---|---|---|
| Optimiseur | AdamW | AdamW |
| LR | `1e-3` | `1e-3` (tête) / `1e-4` (backbone) |
| Weight decay | `1e-4` | `1e-4` |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Batch size | 64 | 64 |
| Max époques | 25 | 25 |
| Early stopping | patience = 5 (val_loss) | patience = 5 (val_loss) |
| Checkpoint | meilleure val_acc | meilleure val_acc |

**Environnement** : Google Colab · GPU NVIDIA T4 / A100 · CUDA 12.x

---

## Performances

### Tableau comparatif final

| Modèle | Test Accuracy | Macro F1-Score | Paramètres |
|---|---|---|---|
| `ImprovedCNN` (from scratch) | **68.3 %** | 0.652 | 423 046 |
| `ResNet18` (fine-tuning) | **89.9 %** | 0.900 | 11 309 382 |

### Analyse par classe — ResNet18

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| buildings | ~0.88 | ~0.87 | ~0.88 |
| forest | ~0.97 | ~0.98 | ~0.97 |
| glacier | ~0.88 | ~0.87 | ~0.87 |
| mountain | ~0.84 | ~0.86 | ~0.85 |
| sea | ~0.94 | ~0.93 | ~0.93 |
| street | ~0.89 | ~0.88 | ~0.88 |

> `forest` et `sea` sont les classes les plus faciles à discriminer.  
> `glacier` / `mountain` restent les plus difficiles (frontière visuelle floue).

### Progression itérative (drafts → final)

| Version | Test Accuracy | Problème principal |
|---|---|---|
| Draft 1 — SimpleCNN sans BatchNorm | ~33 % | Overfitting massif (train 96 % / val 33 %) |
| Draft 2 — LR = 0.1 (SGD) | ~16.7 % | Divergence → loss NaN dès l'époque 4 |
| Draft 3 — sans augmentation | ~63.8 % | Overfitting modéré, plafond de généralisation |
| **Final — ResNet18 fine-tuning** | **~89.9 %** | ✓ |

---

## Structure du projet

```
.
├── Image_Classification_FINAL.ipynb          # Notebook final (version aboutie)
├── Image_Classification_Draft1_NoBatchNorm.ipynb   # Draft 1 — sans BatchNorm
├── Image_Classification_Draft2_HighLR.ipynb        # Draft 2 — LR trop élevé
├── Image_Classification_Draft3_NoAugmentation.ipynb # Draft 3 — sans augmentation
├── README.md                                 # Ce fichier
└── logs/
    ├── ImprovedCNN/                          # Métriques CSV (PyTorch Lightning)
    └── ResNet18_FineTune/
```

---

## Comment exécuter

### Prérequis

- Compte **Google Colab** (GPU recommandé : T4 minimum)
- Compte **Google Drive** avec le dataset Intel téléchargé

### 1. Préparer le dataset

Télécharger le dataset depuis Kaggle et le déposer dans Google Drive à l'emplacement suivant :

```
Mon Drive/
└── deep_learning/
    └── image_dataset/
        ├── seg_train/
        │   └── seg_train/
        │       ├── buildings/
        │       ├── forest/
        │       ├── glacier/
        │       ├── mountain/
        │       ├── sea/
        │       └── street/
        └── seg_test/
            └── seg_test/
                └── (mêmes classes)
```

> Le notebook gère un fallback automatique si la structure est `seg_train/` sans double imbrication.

### 2. Ouvrir le notebook dans Colab

```
Fichier → Ouvrir un notebook → Google Drive → Image_Classification_FINAL.ipynb
```

Ou via l'URL directe si le fichier est partagé :

```
https://colab.research.google.com/drive/<ID_DU_FICHIER>
```

### 3. Activer le GPU

```
Environnement d'exécution → Modifier le type d'exécution → Accélérateur matériel → GPU
```

### 4. Exécuter les cellules

Lancer toutes les cellules dans l'ordre :

```
Exécution → Tout exécuter   (Ctrl + F9)
```

Le notebook monte automatiquement Google Drive et copie les données vers le SSD local de Colab pour accélérer l'I/O.

### Durée estimée

| Étape | Durée estimée |
|---|---|
| Copie des données (Drive → SSD) | 1 – 3 min |
| Entraînement `ImprovedCNN` | 8 – 12 min |
| Entraînement `ResNet18` | 10 – 15 min |
| Évaluation & visualisations | < 2 min |
| **Total** | **~25 – 35 min** |

---

## Dépendances

Toutes les dépendances sont installées automatiquement dans la première cellule du notebook :

```bash
pip install torch torchvision torchmetrics pytorch_lightning \
            scikit-learn seaborn matplotlib pandas
```

| Package | Rôle |
|---|---|
| `torch` / `torchvision` | Framework deep learning, transformations d'images |
| `pytorch_lightning` | Abstraction de la boucle d'entraînement |
| `torchmetrics` | Métriques (accuracy, F1) |
| `scikit-learn` | Matrice de confusion, rapport de classification, courbes ROC |
| `matplotlib` / `seaborn` | Visualisations |
| `pandas` | Tableau comparatif final |

**Versions testées** : Python 3.10 · PyTorch 2.1+ · CUDA 12.x · Google Colab (mai 2026)

---

<div align="center">

Projet réalisé dans le cadre d'un cours de Deep Learning  
Architecture : PyTorch Lightning · ResNet18 · Intel Image Classification Dataset

</div>
