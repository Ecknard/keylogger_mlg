# AI-Driven Keylogger — TP1 Intelligence Artificielle et Cybersecurite


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

Le dashboard lit directement les fichiers JSON sans aucun cache. Il se rafraichit toutes les N secondes (configurable dans la barre laterale) via `st.rerun()`. Un indicateur de fraicheur signale si le keylogger est actif ou inactif.

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

### Tache 4 — Question 4.2 — Bloc try/except

Les touches speciales (Ctrl, Alt, F1...) sont des instances de `pynput.keyboard.Key` et n'ont pas d'attribut `.char`. Tenter d'acceder a `key.char` leve une `AttributeError`. Le bloc `try/except AttributeError` permet de distinguer les deux cas :

```python
try:
    char = key.char       # Touche alphanumérique : .char existe
except AttributeError:
    pass                  # Touche speciale : traitement separe
```

#### Correctif clavier AZERTY (v2)

Sur un clavier AZERTY, appuyer sur la touche physique `0` sans Shift produit le caractere `a` accent grave, la touche `9` produit `c` cedille, et ainsi de suite. pynput retourne `key.char` correspondant au caractere non-shifte, ce qui rendait illisibles les numeros de telephone et les adresses email saisies sans Shift.

Correction appliquee dans `keylogger.py` : utilisation de `key.vk` (virtual key code, independant du layout clavier). Les touches physiques 0 a 9 ont toujours `vk = 48` a `57`. Si `vk` est dans cette plage et que `key.char` n'est pas deja un chiffre (cas Shift enfonce), le chiffre correct est substitue. Les caracteres AltGr (`@`, `#`, `{`) sont preserves.

---

### Tache 5 — Questions

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

#### Question 5.4 — Points faibles et ameliorations apportees

| Dimension | Point faible initial | Amelioration apportee |
|-----------|---------------------|----------------------|
| Encodage clavier | Chiffres AZERTY stockes comme caracteres speciaux | Correction via `key.vk` dans `processkeys()` |
| Pipeline IA | `report()` n'appelait aucun analyseur | Appel de `sentiment_analyzer` et `sensitive_detector` a chaque flush |
| Dashboard | Cache `@st.cache_data` bloquait la lecture des nouveaux fichiers | Lecture directe sans cache, `st.rerun()` controle |
| Contexte | Aucune information sur l'application active | Extension B (`app_context.py`) |
| Securite log | Fichier en clair sur le disque | Chiffrement AES-256-GCM (Extension C) |

---

### Tache 6 — Moteur de sentiment multilingue (v4)

#### Choix architectural : Character N-gram TF-IDF + Regression Logistique

La version initiale utilisait VADER, un dictionnaire anglophone. Elle ne reconnaissait aucune phrase francaise negative : `"je suis en detresse"`, `"c est penible"`, `"quel cata"` retournaient toutes un score de 0.000, produisant une ligne plate dans le dashboard.

La version 4 abandonne completement l'approche par dictionnaire au profit d'un modele d'apprentissage automatique entraine sur un corpus multilingue.

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

**Entraînement automatique :**

Le modele est entraine au premier lancement et serialise dans `data/sentiment_model.joblib`. Les executions suivantes chargent directement le modele depuis le disque. Aucun telechargement reseau n'est necessaire.

**Cas limite : textes tres courts**

Les textes de moins de 2 mots sont classes `trop_court` et ne sont pas sauvegardes dans `sentiments.json`. Cette valeur minimale (inferieure aux 3 mots de la version precedente) permet de capturer des saisies courtes frequentes en contexte clavier.

---

### Tache 7 — Detection d'anomalies comportementales

**Collecte :** 30 a 60 minutes de frappe normale sont necessaires pour etablir un profil de reference fiable.

**Normalisation :** les features ont des echelles tres differentes (delais en secondes, ratios entre 0 et 1). Sans normalisation par `StandardScaler`, les features a grande echelle domineraient le calcul de distance et biaiseraient le modele.

**Parametre de contamination :** valeur initiale de 0.05 (5 %). A ajuster a la baisse si le taux de faux positifs observes est trop eleve en production.

---

### Tache 8 — Classification de donnees sensibles

#### Approche hybride : Regex valide + Random Forest

**Regex avec post-validation :**

| Pattern | Validation supplementaire |
|---------|--------------------------|
| Carte bancaire | Algorithme de Luhn (elimine les faux positifs) |
| Numero de securite sociale | Verification de la cle INSEE |
| Telephone francais | Couverture etendue a tous les indicatifs 0[1-9], `+33`, `0033` |
| Email | Regex robuste avec sous-domaines et TLDs longs |
| JWT, cle API, IBAN, IPv4 | Patterns specifiques sans validation supplementaire |

**Redaction intelligente :**

Les valeurs ne sont jamais stockees en clair dans `detections.json`. Seul leur hache SHA-256 et leur longueur sont conserves. La redaction preserve le format lisible (`alice@***.com`, `**** **** **** 9012`, `** ** ** ** 78`).

**Choix Random Forest :** gere naturellement les features mixtes (continues et binaires), robuste aux valeurs aberrantes, permet d'inspecter `feature_importances_`. La metriques prioritaire est le rappel (recall) : un faux negatif (donnee sensible non detectee) est bien plus couteux qu'un faux positif.

---

### Tache 9 — Visualisations (dashboard Streamlit)

Le dashboard (`extension/dashboard.py`) propose cinq vues :

| Vue | Contenu |
|-----|---------|
| Vue globale | Timeline des sentiments, heatmap horaire, alertes, donnees sensibles, log brut, delais inter-touches |
| Sentiments | Timeline detaillee, donut de repartition, distribution par langue, statistiques par langue |
| Anomalies | Scatter des alertes, histogramme des delais |
| Donnees sensibles | Donut des types detectes, liste des detections recentes |
| Logs bruts | Affichage paginable du fichier `log.txt` |

**Absence de cache :** la fonction `load_all()` lit directement les fichiers JSON a chaque cycle de rafraichissement. L'ancien `@st.cache_data(ttl=3)` qui bloquait la mise a jour des donnees a ete supprime.

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
  → lecture directe des fichiers JSON sans cache
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
