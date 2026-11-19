# Anechoics : optimiser le placement de l'absorbant dans une chambre anéchoïque

Projet d'Enseignement d'Intégration (EI ST5, « Contrôle de la pollution électromagnétique »), CentraleSupélec, novembre 2026. Projet réalisé en une semaine, en équipe de six.

## L'idée

Une chambre anéchoïque sert à tester les appareils électroniques sans que les échos sur les murs faussent la mesure. Aujourd'hui, on couvre tous les murs de mousse absorbante, de la même façon. C'est cher, et chaque rénovation immobilise la chambre.

On s'est demandé si on pouvait mettre **moins de mousse, mais au bon endroit**, sans dégrader la chambre. On a donc modélisé la chambre, puis optimisé la répartition de l'absorbant sur le mur, à quantité de matériau fixée.

## Résultats (en simulation)

- **À une seule fréquence**, l'énergie baisse de 69 %… mais le pic de résonance se décale juste à côté. Optimiser une fréquence ne suffit pas.
- **Sur toute la bande**, en visant le pire pic, l'énergie maximale passe de 294,3 à 13,8 : **−95 %**, avec 40 % du mur couvert.
- **Avec un mur en créneaux** (fractale de Koch, niveau 2), l'énergie baisse encore de 31 % par rapport à un mur plan, à quantité de mousse égale (à une fréquence donnée).

Le détail est dans le [rapport](Rapport_Groupe_3.pdf).

## Comment ça marche

1. **Le modèle** : équation de Helmholtz en 2D, résolue par différences finies (schéma à 5 points). La source est en haut, le mur à optimiser en bas, et la mousse est représentée par une condition de Robin dont le coefficient vient d'un absorbant réel (Eccosorb LS-26).
2. **Le critère** : l'énergie du champ dans la chambre, comptée double autour de l'appareil testé.
3. **L'optimisation** : descente de gradient projeté. Le gradient est calculé par la méthode adjointe (deux résolutions suffisent pour tout le mur), puis une projection garde la quantité de mousse constante.
4. **Toute la bande** : on minimise une norme p (p = 10) des énergies sur 191 fréquences, une sorte de maximum « lissé » qu'on peut dériver.
5. **La vraie pose** : la densité optimisée (entre 0 et 1) est transformée en « panneau ou rien », par seuillage ou par motif alterné.

## Organisation du code

| Fichier | Rôle |
|---|---|
| `_env.py` | Types des nœuds de la grille (air, source, murs réfléchissants, mur absorbant) |
| `preprocessing.py` | Construction de la grille, du mur (plan ou fractal) et de la répartition de départ |
| `processing.py` | Solveur de Helmholtz par différences finies |
| `postprocessing.py` | Figures de base |
| `compute_alpha.py` | Coefficient de la mousse à partir des propriétés du matériau |
| `demo_control_polycopie2024.py` | Énergie, gradient adjoint, projection, optimisation (une fréquence ou toute la bande), binarisation |
| `etude_fractale.py` | Comparaison mur plan / murs fractals à quantité de mousse égale |

## Lancer le code

Python 3 avec NumPy, SciPy et Matplotlib :

```bash
pip install -r requirements.txt
```

**Optimisation sur toute la bande** (réglages par défaut, quelques minutes) :

```bash
python demo_control_polycopie2024.py
```

Pour optimiser à une seule fréquence, mettre `OPTIMISER_MULTI_FREQ = False` dans le programme principal.

**Étude des murs fractals** (une quinzaine de minutes ; résultats et figures dans `figures_fractales/`) :

```bash
python etude_fractale.py
```

## Limites

Ce sont des résultats de simulation : modèle 2D, un seul mur optimisé, absorbant sans épaisseur, et une descente de gradient qui ne garantit qu'un optimum local. La suite naturelle serait de passer en 3D, d'optimiser toutes les parois et de valider par la mesure.

## Contexte

Le solveur de base (`processing.py`, `preprocessing.py`, `postprocessing.py`) est fourni par l'équipe pédagogique de CentraleSupélec. Le reste du code, la modélisation et les résultats sont le travail de l'équipe.
