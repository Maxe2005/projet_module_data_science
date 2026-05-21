## Instalation

### venv

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

### Configuration

Vous pouvez modifier les paramètres en haut du fichier `main.py` ou en utilisant des variables d'environnement.
exemple:

```python
CLEANING_TYPE = CleaningType.REMOVE
COLUMNS_TO_REMOVE = ["id", "children", "education", "married", "postal_code"]
```

### Actions

Vous pouvez modifier l'action effectuée par le script en commentant/décommentant les différentes fonctions dans actions dans `main.py`:

```python
def actions(data):
    # simple_train_validate(data)

    cross_validate(data)

    # compare_classifiers(file_path_out)

    evaluate_saved_model(model_path="models/logistic_regression_cv_best.pkl", data=data)
```

### Logs

L'utilisation de `cross_validate` génère des logs dans le dossier 'logs/' avec des noms de fichiers basés sur la date et l'heure d'exécution.

### Données prétraitées

Les données prétraitées sont enregistrées au format CSV dans un fichier nommé `car_insurance_formatted.csv` dans le même dossier que les données d'origine. Vous pouvez modifier ce nom et son emplacement en changeant la variable `file_path_out` dans `main.py`.

### Correlations

Pour analyser les corrélations, vous pouvez utiliser le script `correlations/correlations.py` qui génère des graphiques de corrélation pour les variables numériques et catégorielles. Les graphiques sont enregistrés dans le dossier `correlations/`.

Attention : L'analyse se base sur les données prétraitées, donc assurez-vous d'exécuter `preprocess_data` (ou tout simplement le script `main.py`) avant d'utiliser ce script.

### Visualisation des données

Vous trouverez des outils de visualisation dans le fichier `visualization_utils.py`


## Linter et formateur

Le projet contient Black (formateur), isort (tri des imports), et Flake8 (linter), plus une configuration `pre-commit`.

Commandes utiles:

```bash
# installer les dépendances (si pas déjà fait)
pip install -r requirements.txt

# installer les hooks pre-commit
pre-commit install

# formater tous les fichiers suivis par git
pre-commit run --all-files

# ou exécuter black/flake8 manuellement
black .
flake8
```

VS Code: la config recommandée est dans `.vscode/settings.json` pour utiliser Black et Flake8.


