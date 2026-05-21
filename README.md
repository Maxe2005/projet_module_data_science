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


