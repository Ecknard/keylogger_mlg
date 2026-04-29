# AI-Driven Keylogger — TP1 Intelligence Artificielle et Cybersecurite

https://github.com/Ecknard/keylogger_mlg.git

---

## Avertissement ethique et legal

Ce projet est exclusivement pedagogique. L'utilisation d'un keylogger sans le consentement explicite et ecrit de la personne surveillee est illegale en France (Loi Godfrain, article 323-1 du Code penal) et dans l'Union Europeenne (RGPD, directive NIS2). Les sanctions encourues vont jusqu'a deux ans d'emprisonnement et 60 000 euros d'amende.

Ce code ne doit etre execute que sur votre propre machine, dans un environnement de test isole.

---

## Structure du projet

```
ai_keylogger/
├── keylogger.py            Partie I   : Capture des frappes clavier
├── sentiment_analyzer.py   Partie II  : Analyse de sentiments multilingue (ML)
├── anomaly_detector.py     Partie II  : Detection d'anomalies (Isolation Forest)
├── sensitive_detector.py   Partie III : Classification donnees sensibles (Regex + RF)
├── report_generator.py     Partie IV  : Rapports HTML et visualisations Plotly
├── extension/
│   ├── __init__.py
│   ├── app_context.py      Extension B : Contexte applicatif (application active)
│   ├── dashboard.py        Extension D : Dashboard Streamlit temps reel
│   └── encryption.py       Extension C : Chiffrement AES-256-GCM
├── data/
│   ├── log.txt             Frappes capturees (mode append)
│   ├── sentiments.json     Resultats d'analyse de sentiment
│   ├── alerts.json         Alertes anomalies comportementales
│   ├── detections.json     Donnees sensibles detectees (valeurs hachees)
│   ├── metadata.json       Metadonnees de frappe (delais, types de touches)
│   ├── sentiment_model.joblib   Modele ML sentiment (genere automatiquement)
│   └── report.html         Rapport HTML exporte
├── tests/
│   └── test_all.py         Tests unitaires (pytest)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Creer et activer l'environnement virtuel
python -m venv env

# Windows
.\env\Scripts\activate

# Linux / macOS
source env/bin/activate

# 2. Installer les dependances
pip install -r requirements.txt

# 3. Verifier l'installation
python -c "import pynput; print('pynput OK')"
python -c "import sklearn; print('sklearn OK')"
python -c "import langdetect; print('langdetect OK')"
python -c "import streamlit; print('streamlit OK')"
```

### Dependances principales

| Paquet | Version minimale | Role |
|--------|-----------------|------|
| pynput | 1.7.6 | Capture des evenements clavier |
| scikit-learn | 1.4.0 | Modele de sentiment (TF-IDF + LR) et detection de mots de passe (Random Forest) |
| langdetect | 1.0.9 | Detection automatique de la langue (FR/EN/ES/DE/IT) |
| numpy | 1.26.0 | Operations vectorielles |
| joblib | 1.3.0 | Serialisation des modeles ML |
| streamlit | 1.32.0 | Dashboard de supervision |
| plotly | 5.18.0 | Visualisations interactives |
| cryptography | 42.0.0 | Chiffrement AES-256-GCM des logs |

---

## Utilisation

### Lancer le keylogger avec le pipeline IA temps reel

```bash
python keylogger.py
```

A chaque intervalle de flush (10 secondes par defaut), le keylogger :
- ecrit dans `data/log.txt` en mode append,
- appelle `sentiment_analyzer` qui met a jour `data/sentiments.json`,
- appelle `sensitive_detector` qui met a jour `data/detections.json`,
- sauvegarde les metadonnees de frappe dans `data/metadata.json`.

Le pipeline IA tourne dans des threads daemon non bloquants : la capture des frappes n'est jamais interrompue.

```bash
# Arreter avec Ctrl+C
```

### Lancer le dashboard de supervision

```bash
streamlit run extension/dashboard.py
# Ouvrir http://localhost:8501
```

Le dashboard lit directement les fichiers JSON et se rafraichit toutes les N secondes (configurable dans la barre laterale) via `st.rerun()`. Un indicateur de fraicheur signale si le keylogger est actif ou inactif.

### Tester les modules independamment

```bash
# Analyse de sentiment
python sentiment_analyzer.py

# Detection de donnees sensibles
python sensitive_detector.py

# Detection d'anomalies
python anomaly_detector.py

# Rapport HTML
python report_generator.py
```

### Tests unitaires

```bash
python -m pytest tests/ -v
```

---

## Reponses aux questions du TP

### Tache 1 — Analyse conceptuelle

#### Question 1.1 — Definition et dangers

Un keylogger est un programme qui intercepte et enregistre les evenements clavier d'un systeme a l'insu de l'utilisateur.

| Axe | Danger concret |
|-----|---------------|
| Donnees visees | Mots de passe, coordonnees bancaires, messages prives, identifiants professionnels |
| Vecteurs d'infection | Email de phishing, logiciel pirate, cle USB infectee, exploit navigateur |
| Persistance | Cle de registre (HKCU\Run), cron job, service systeme, rootkit |
| Exfiltration | Email SMTP automatique, upload HTTP/FTP, webhook Discord/Telegram |

#### Question 1.2 — Usages legitimes

| Secteur | Usage | Condition legale |
|---------|-------|-----------------|
| Entreprise | Audit securite interne, investigation d'incident | Consentement ecrit, charte informatique, information des employes (RGPD art. 13) |
| Parentalite | Controle parental enfant mineur | Autorite parentale legale, enfant sous 15 ans |
| Forensic / SOC | Investigation post-incident | Mandat judiciaire ou autorisation hierarchique documentee |

---

### Tache 3 — Questions d'analyse de code

#### Question 3.2 — Le parametre `on_press`

`on_press` est un callback (fonction de rappel) passe au Listener. pynput l'appelle automatiquement dans son thread interne a chaque evenement clavier. La valeur passee est une reference a la fonction (sans parentheses) : `on_press=processkeys`.

#### Question 3.3 — `with` et `join()`

`with keyboard_listener:` — le gestionnaire de contexte appelle automatiquement `listener.start()` a l'entree et `listener.stop()` a la sortie, meme en cas d'exception.

`keyboard_listener.join()` — bloque le thread principal jusqu'a ce que le listener s'arrete. Sans `join()`, le programme se terminerait immediatement apres la ligne `with`, arretant le listener avant toute capture.

---

### Tache 4 — Capture et traitement des touches

#### Question 4.2 — Bloc try/except

Les touches speciales (Ctrl, Alt, F1...) sont des instances de `pynput.keyboard.Key` et n'ont pas d'attribut `.char`. Tenter d'acceder a `key.char` leve une `AttributeError`. Le bloc `try/except AttributeError` permet de distinguer les deux cas :

```python
try:
    char = key.char       # Touche alphanumérique : .char existe
except AttributeError:
    pass                  # Touche speciale : traitement separe
```

#### Gestion du clavier AZERTY

Sur un clavier AZERTY, le caractere non-shifte des touches numeriques est `a` accent grave, `&`, `e` accent aigu, `"`, `'`, `(`, `-`, `e` accent grave, `_`, `c` cedille. Pour que les chiffres saisis soient correctement enregistres, `processkeys()` s'appuie sur `key.vk` (virtual key code) qui est independant du layout clavier. Les touches physiques `0` a `9` ont toujours `vk` dans la plage 48-57 (rangee du haut) ou 96-105 (pave numerique), ce qui permet de substituer le bon chiffre. Les caracteres AltGr (`@`, `#`, `{`) sont preserves.

---

### Tache 5 — Persistence dans un fichier log

#### Question 5.2 — Modes de fichier

| Mode | Comportement |
|------|-------------|
| `'a'` | Append : ajoute a la fin sans ecraser. Cree le fichier si inexistant. |
| `'w'` | Write : ecrase tout le contenu existant. |

Le mode `'a'` est utilise pour ne pas perdre les logs precedents entre deux appels a `report()`. Le buffer `log` est reinitialise apres chaque ecriture pour eviter la duplication des donnees. Le fichier est ferme apres chaque ecriture pour vider le buffer OS, liberer le descripteur de fichier et garantir qu'un autre processus peut lire le fichier immediatement.

#### Question 5.3 — Timer auto-relance

```python
def report(interval=10):
    # ecriture du log
    timer = threading.Timer(interval, report, args=[interval])
    timer.daemon = True
    timer.start()  # se relance lui-meme indefiniment, sans bloquer
```

#### Question 5.4 — Points faibles du keylogger de base et solutions apportees

| Dimension | Point faible | Solution dans ce projet |
|-----------|-------------|------------------------|
| Pipeline IA | Un keylogger brut ne fait aucune analyse | `report()` appelle `sentiment_analyzer` et `sensitive_detector` a chaque flush |
| Thread-safety | Acces concurrents au buffer entre capture et flush | Verrou `threading.Lock` sur `log` et `keystroke_metadata` |
| Contexte | Aucune information sur l'application active | Extension B (`app_context.py`) |
| Securite du log | Fichier en clair sur le disque | Chiffrement AES-256-GCM (Extension C) |
| Encodage clavier | Mauvaise interpretation des chiffres sur AZERTY | Utilisation de `key.vk` dans `processkeys()` |

---

### Tache 6 — Moteur de sentiment multilingue

#### Choix de l'approche : Character N-gram TF-IDF + Regression Logistique

Le sujet propose plusieurs bibliotheques pour l'analyse de sentiments : VADER, TextBlob, Transformers HuggingFace, CamemBERT/FlauBERT. Le choix effectue s'appuie sur une analyse des contraintes reelles du projet.

**Justification du choix vis-a-vis des approches proposees :**

| Approche proposee | Contrainte identifiee | Decision |
|------------------|----------------------|----------|
| VADER (vaderSentiment) | Dictionnaire anglophone uniquement, incapable de traiter les frappes francaises qui constituent la majorite du flux attendu | Ecartee |
| TextBlob | Necessite une traduction intermediaire pour chaque langue, latence et dependance reseau incompatibles avec le temps reel | Ecartee |
| Transformers HuggingFace | Modeles de plusieurs centaines de Mo, chargement long, empreinte memoire elevee, GPU recommande | Ecartee |
| CamemBERT / FlauBERT | Excellent en francais mais mono-langue, requiert un fine-tuning specifique | Ecartee |
| TF-IDF char n-grams + Logistic Regression | Multilingue par nature (apprentissage morphologique), modele de moins de 1 Mo, inference en millisecondes, aucun telechargement externe | Retenue |

**Pipeline ML :**

```
Texte brut
  → _clean()                     suppression timestamps, balises clavier
  → langdetect.detect_langs()    detection de langue + drapeau
  → TfidfVectorizer(
        analyzer='char_wb',      n-grammes de caracteres
        ngram_range=(2, 5),      de 2 a 5 caracteres par token
        sublinear_tf=True        TF logarithmique
    )
  → LogisticRegression(
        C=3.0,
        class_weight='balanced'
    )
  → prob_pos dans [0, 1]
  → tanh(logit(prob_pos))        score continu dans [-1, +1]
  → seuils                       label sur 5 niveaux
```

**Pourquoi les n-grammes de caracteres fonctionnent sans lexique :**

Les sequences de 2 a 5 caracteres capturent automatiquement les patterns morphologiques porteurs de sentiment, independamment de la langue. Le prefixe `"horr"` apparait dans `horrible` (francais/anglais), `horrible` (espagnol), `orribile` (italien). Le radical `"frust"` est commun a `frustre`, `frustrated`, `frustriert`, `frustrato`. Le modele apprend ces correspondances par entrainement, sans qu'on lui indique explicitement le sens des mots.

**Performances sur le corpus de validation :**

| Metrique | Valeur |
|----------|--------|
| F1 weighted (CV 5-fold) | 0.950 +/- 0.018 |
| Taille du corpus | 280 exemples (140 negatifs / 140 positifs) |
| Langues couvertes | FR, EN, ES, DE, IT |

**Sortie de `analyze_sentiment()` :**

| Champ | Type | Description |
|-------|------|-------------|
| `score` | float [-1, +1] | Score continu calcule via tanh(logit) |
| `label` | str | `tres_positif`, `positif`, `neutre`, `negatif`, `tres_negatif`, `trop_court` |
| `language` | str | Code ISO detecte par langdetect |
| `lang_flag` | str | Drapeau emoji de la langue |
| `lang_conf` | float | Confiance de la detection de langue |
| `confidence` | float | Confiance de l'analyse sentiment |
| `prob_pos` | float | Probabilite brute classe positive |

**Entrainement automatique :**

Le modele est entraine au premier lancement et serialise dans `data/sentiment_model.joblib`. Les executions suivantes chargent directement le modele depuis le disque. Aucun telechargement reseau n'est necessaire.

**Cas limite : textes tres courts**

Les textes de moins de 2 mots sont classes `trop_court` et ne sont pas sauvegardes dans `sentiments.json`.

---

### Tache 7 — Detection d'anomalies comportementales

#### Choix de l'algorithme : Isolation Forest

Le sujet propose quatre algorithmes de detection d'anomalies : Isolation Forest, One-Class SVM, Autoencoder, Local Outlier Factor (LOF).

**Justification du choix vis-a-vis des approches proposees :**

| Algorithme propose | Analyse | Decision |
|-------------------|---------|----------|
| One-Class SVM | Sensible au choix du noyau et des hyperparametres, cout quadratique sur les grands jeux de donnees | Ecartee |
| Autoencoder | Requiert TensorFlow/Keras, entrainement long, surdimensionne pour 8 features | Ecartee |
| LOF | Adapte aux anomalies locales, mais cout O(n²), difficile en temps reel sur une fenetre glissante | Ecartee |
| Isolation Forest | Lineaire en temps d'entrainement, pas de noyau a choisir, contamination parametrable, excellent sur donnees multidimensionnelles | Retenue |

**Pipeline :**

```
Collecte de 100+ evenements de frappe
  → extract_features() sur fenetres glissantes de 20 frappes
  → 8 features : delais (min/max/mean/std/median), ratios de types de touches, burst_count
  → StandardScaler (obligatoire car echelles tres differentes)
  → IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
  → predict() : -1 = anomalie, 1 = normal
  → si anomalie detectee → alerts.json (append)
```

**Parametres retenus :**

- **Collecte :** 30 a 60 minutes de frappe normale pour etablir un profil de reference fiable. Minimum absolu fixe a 100 evenements (`MIN_SAMPLES_TRAIN`).
- **Normalisation :** `StandardScaler` obligatoire car les features ont des echelles tres differentes (delais en secondes entre 0 et plusieurs secondes, ratios entre 0 et 1). Sans normalisation, les delais domineraient le calcul de distance.
- **Contamination :** 0.05 (5 %) en valeur par defaut. A ajuster a la baisse si le taux de faux positifs observes est trop eleve en production.
- **Fenetre glissante :** 20 frappes consecutives (`WINDOW_SIZE`), verifiees toutes les 5 secondes par le thread daemon `AnomalyMonitor`.

---

### Tache 8 — Classification de donnees sensibles

#### Approche hybride : Regex validees + Random Forest

Le sujet propose de combiner detection par expressions regulieres et classification ML. Les deux approches sont complementaires dans ce projet.

**Couche 1 : Regex avec post-validation**

| Pattern | Validation supplementaire |
|---------|--------------------------|
| Carte bancaire | Algorithme de Luhn (elimine les faux positifs mathematiquement) |
| Numero de securite sociale | Verification de la cle INSEE |
| Telephone francais | Couverture de tous les indicatifs 0[1-9], `+33`, `0033` |
| Email | Regex robuste avec sous-domaines et TLDs longs |
| JWT, cle API, IBAN, IPv4 | Patterns specifiques |

La validation algorithmique (Luhn, cle INSEE) est essentielle : sans elle, n'importe quelle suite de 16 chiffres serait classee comme carte bancaire, ce qui produirait un taux de faux positifs inacceptable.

**Couche 2 : Classification ML pour les mots de passe**

Le sujet propose trois classifieurs : Naive Bayes, Random Forest, SVM.

| Classifieur propose | Analyse | Decision |
|--------------------|---------|----------|
| Naive Bayes | Suppose l'independance des features, ce qui est faux pour nos features (longueur et entropie sont correlees) | Ecarte |
| SVM | Sensible au choix du noyau et de C, moins interpretable, pas d'acces direct aux feature importances | Ecarte |
| Random Forest | Gere naturellement les features mixtes (continues et binaires), robuste aux valeurs aberrantes, permet d'inspecter `feature_importances_`, class_weight configurable | Retenu |

Features extraites pour chaque token candidat : longueur, entropie de Shannon, ratio majuscules/minuscules/chiffres/speciaux, presence simultanee des trois classes de caracteres.

**Metrique prioritaire : le rappel (recall)**

Un faux negatif (donnee sensible non detectee et donc stockee en clair) est bien plus couteux qu'un faux positif (valeur banale masquee par erreur). Le seuil de classification est donc ajuste pour maximiser le rappel, quitte a sacrifier un peu de precision.

**Redaction intelligente :**

Les valeurs detectees ne sont jamais stockees en clair dans `detections.json`. Seul leur hache SHA-256 et leur longueur sont conserves. La redaction dans les logs preserve le format lisible (`alice@***.com`, `**** **** **** 9012`, `** ** ** ** 78`) pour faciliter l'audit sans exposer les donnees reelles.

---

### Tache 9 — Visualisations (dashboard Streamlit)

#### Choix de la bibliotheque : Streamlit + Plotly

Le sujet propose cinq bibliotheques : Matplotlib, Seaborn, Plotly, Dash, Streamlit.

**Justification du choix :**

| Bibliotheque proposee | Analyse | Decision |
|----------------------|---------|----------|
| Matplotlib / Seaborn | Graphiques statiques, aucune interactivite, rendu web complexe | Ecartees |
| Dash | Tres puissant mais architecture complexe (callbacks, layouts), verbeux pour un dashboard de supervision | Ecartee |
| Plotly (seul) | Graphiques interactifs mais manque l'ossature applicative (widgets, layout, rafraichissement) | Integre a Streamlit |
| Streamlit + Plotly | Streamlit fournit les widgets et le rafraichissement, Plotly fournit les graphiques interactifs. Combinaison rapide a developper et naturelle pour un dashboard de supervision | Retenue |

**Cinq vues implementees :**

| Vue | Contenu |
|-----|---------|
| Vue globale | Timeline des sentiments, heatmap horaire, alertes, donnees sensibles, log brut, delais inter-touches |
| Sentiments | Timeline detaillee, donut de repartition, distribution par langue, statistiques par langue |
| Anomalies | Scatter des alertes, histogramme des delais |
| Donnees sensibles | Donut des types detectes, liste des detections recentes |
| Logs bruts | Affichage paginable du fichier `log.txt` |

**Rafraichissement temps reel :**

La fonction `load_all()` lit directement les fichiers JSON a chaque cycle de rafraichissement. Le dashboard utilise `st.rerun()` pour re-executer integralement le script toutes les N secondes, garantissant que les nouvelles donnees capturees apparaissent immediatement.

---

### Tache 10 — Generation automatisee de rapports

#### Format retenu : HTML genere par Jinja2 + Plotly

Le sujet propose quatre formats : HTML, PDF, Markdown+pandoc, Notebook Jupyter.

**Justification :**

| Format propose | Analyse | Decision |
|----------------|---------|----------|
| PDF (reportlab/weasyprint) | Standard professionnel mais perte de l'interactivite des graphiques | Ecarte |
| Markdown + pandoc | Necessite une chaine d'outils externe (pandoc), pas adapte aux graphiques interactifs | Ecarte |
| Notebook Jupyter | Bon pour la reproductibilite mais lourd a partager, necessite Jupyter pour la lecture | Ecarte |
| HTML (Jinja2 + Plotly) | Interactif, auto-contenu, lisible dans n'importe quel navigateur sans installation | Retenu |

La fonction `generate_html_report()` charge les donnees JSON, construit les graphiques Plotly, et rend le tout via un template Jinja2 dans `data/report.html`.

---

### Partie V — Extensions implementees

| Extension | Module | Description |
|-----------|--------|-------------|
| B | `extension/app_context.py` | Capture de l'application active (Windows/Linux/macOS) avec fallback `pygetwindow` |
| C | `extension/encryption.py` | Chiffrement AES-256-GCM avec derivation de cle PBKDF2-SHA256 (480 000 iterations, OWASP 2023) |
| D | `extension/dashboard.py` | Dashboard Streamlit temps reel avec rafraichissement via `st.rerun()` |

---

## Architecture du pipeline temps reel

```
Frappes clavier
  → processkeys()          correction AZERTY (key.vk), metadonnees
       |
  [toutes les 10s]
       |
  report()                 flush thread-safe du buffer
       |
       +--→ log.txt        ecriture en mode append
       |
       +--→ sentiment_analyzer.analyze_sentences_from_log()
       |         → sentiments.json (append)
       |
       +--→ sensitive_detector.analyze_text()
       |         → detections.json (append, valeurs hachees)
       |
       +--→ metadata.json  delais, types de touches (append, max 10 000 entrees)

[dashboard Streamlit]
  → lecture directe des fichiers JSON
  → st.rerun() toutes les N secondes
  → indicateur de fraicheur (age du dernier log.txt)
```

---

## Architecture de securite

```
Frappes → processkeys() → buffer en memoire
                               |
                        [flush 10s]
                               |
                   sensitive_detector → masquage + hachage SHA-256
                               |
                   [optionnel] encrypt_file (AES-256-GCM)
                               |
                           data/log.enc
```

---

## Contribution

Ce projet est une correction pedagogique. Pour toute amelioration :

1. Creer une branche feature (`git checkout -b feature/nom-de-la-feature`)
2. Commiter les modifications (`git commit -m 'Add: description'`)
3. Pousser la branche (`git push origin feature/nom-de-la-feature`)
4. Ouvrir une Pull Request avec une description claire des changements apportes
