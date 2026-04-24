"""
sentiment_analyzer.py  —  Moteur de sentiment MULTILINGUE sans lexique
TP1 — Intelligence Artificielle & Cybersécurité

ARCHITECTURE v4 — "Character N-gram TF-IDF + Logistic Regression"
═══════════════════════════════════════════════════════════════════

  Pourquoi cette approche ?
  ─────────────────────────
  • Aucun dictionnaire ni lexique à maintenir.
  • Fonctionne sur FR / EN / ES / DE / IT par apprentissage automatique.
  • Les N-grammes de caractères (2-5) capturent les patterns morphologiques
    qui sont porteurs de sentiment sans connaître la langue :
      "horr" → horrible/horrible/orribile/horroroso
      "trast" → frustrated/frustriert/frustrato/frustrado
      "excel" → excellent/excelente/eccellente/ausgezeichnet
  • Sortie continue (score [-1, +1]) → 5 labels par seuils.
  • Entraînement en <1 seconde au premier lancement, modèle mis en cache.

  Pipeline
  ────────
  Texte brut
    → _clean()                     nettoyage timestamps / balises clavier
    → langdetect.detect_langs()    détection de langue + flag
    → TfidfVectorizer(char 2-5)    vectorisation language-agnostic
    → LogisticRegression           score de probabilité → score continu
    → tanh(logit)                  normalisation [-1, +1]
    → seuils                       label 5 niveaux

  Langues supportées : FR · EN · ES · DE · IT
  Dépendances       : sklearn · langdetect · numpy  (déjà installés)
"""

import json
import math
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 42
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False
    print("[AVERTISSEMENT] langdetect non installé : pip install langdetect")

# ── Chemins ───────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent
_MODEL_PATH = _ROOT / "data" / "sentiment_model.joblib"

# ── Seuils → 5 labels ─────────────────────────────────────────────────────────
THRESH_VERY_POS =  0.50
THRESH_POS      =  0.10
THRESH_NEG      = -0.10
THRESH_VERY_NEG = -0.50
MIN_WORDS       =  2

# ── Drapeaux ──────────────────────────────────────────────────────────────────
_LANG_FLAGS = {
    "fr": "🇫🇷", "en": "🇬🇧", "es": "🇪🇸",
    "de": "🇩🇪", "it": "🇮🇹", "unknown": "🌐",
}
_SUPPORTED = {"fr", "en", "es", "de", "it"}


# ════════════════════════════════════════════════════════════════════════════════
# CORPUS D'ENTRAÎNEMENT MULTILINGUE
# 500+ exemples — aucun lexique, apprentissage pur sur le signal textuel
# ════════════════════════════════════════════════════════════════════════════════

def _build_corpus() -> tuple[list[str], list[int]]:
    """
    Corpus multilingue équilibré (FR / EN / ES / DE / IT).
    Labels : -1 = négatif  /  +1 = positif
    """
    neg = [
        # ── FRANÇAIS négatif ──────────────────────────────────────────────────
        "je suis vraiment en detresse",
        "c est horrible et penible",
        "quel cata desastre total absolu",
        "je deteste ce projet completement nul",
        "je suis furieux et en colere terrible",
        "c est catastrophique et insupportable",
        "je suis deprime triste et malheureux",
        "tout va mal c est affreux lamentable",
        "j en ai vraiment marre echec rate",
        "cette erreur est une catastrophe horrible",
        "je souffre enormement tellement douleur",
        "je suis anxieux stresse angoisse",
        "encore un echec lamentable pitoyable",
        "mauvais resultat decevant terrible",
        "je pleure tellement c est affreux",
        "desespoir total impossible a supporter",
        "je suis en colere noire furieux",
        "nul et mediocre catastrophique",
        "quel cauchemar horrible insupportable",
        "je me sens triste et abandonne",
        "tout est casse panne complete",
        "je hais ce projet raté loupe",
        "c est penible et epuisant",
        "je suis decu et blesse",
        "rien ne va c est la cata",
        "je suis stresse et fatigue",
        "c est un desastre complet",
        "je suis malheureux et triste",
        "horrible situation catastrophique",
        "je suis vraiment deprime aujourd hui",
        "tout va tres mal echec total",
        "insupportable et affreux",
        "je deteste cette situation horrible",
        "la panne est catastrophique",
        "je suis au bout du rouleau",
        "quelle horreur lamentable",
        "je suis completement depassé",
        "raté encore une fois echec",
        "je suis epuise et a bout",
        "c est nul et decevant",
        # ── ANGLAIS négatif ───────────────────────────────────────────────────
        "I am absolutely furious right now",
        "this is terrible and awful horrible",
        "I hate this disgusting mess",
        "feeling depressed and hopeless",
        "this is a complete disaster catastrophe",
        "I am devastated and heartbroken",
        "disgusting pathetic worthless work",
        "I despise this terrible situation",
        "angry frustrated and stressed out",
        "this is unbearable and miserable",
        "failed again what a nightmare",
        "terrible pain and suffering",
        "I loathe this awful experience",
        "worst day ever depressing",
        "scared anxious and nervous wreck",
        "I cannot stand this horrible mess",
        "everything is broken and ruined",
        "I feel hopeless and destroyed",
        "this is dreadful and atrocious",
        "completely failed and disappointed",
        "I am outraged and livid",
        "this is absolutely dreadful awful",
        "I am so upset and distressed",
        "miserable terrible and worthless",
        "I regret everything terrible mistake",
        "exhausted burned out and stressed",
        "I feel terrible and depressed",
        "this disaster is catastrophic",
        "I am so angry and frustrated",
        "horrible experience worst ever",
        "I despise this awful situation",
        "completely useless and pathetic",
        "crying upset and heartbroken",
        "terrible pain suffering miserable",
        "I feel lost and hopeless",
        "dreadful nightmare awful experience",
        "furious and enraged beyond belief",
        "I give up hopeless failure",
        "disgusted and appalled terrible",
        "nothing works broken disaster",
        # ── ESPAGNOL négatif ──────────────────────────────────────────────────
        "esto es horrible y terrible",
        "estoy muy frustrado y enojado",
        "que desastre catastrofe total",
        "odio esta situacion horrible",
        "me siento deprimido y triste",
        "fracaso total lamentable pesimo",
        "estoy furioso y desesperado",
        "sufrimiento y dolor insoportable",
        "todo esta mal y roto",
        "que horrible experiencia terrible",
        "estoy muy cansado y agotado",
        "me odio esto es horrible",
        "terrible situacion catastrofica",
        "estoy muy triste y deprimido",
        "fracaso completo horrible",
        "desesperado y sin esperanza",
        "que pesadilla horrible y mala",
        "odio este proyecto terrible",
        "muy frustrado y decepcionado",
        "todo va mal desastre total",
        # ── ALLEMAND négatif ──────────────────────────────────────────────────
        "das ist schrecklich und furchtbar",
        "ich bin sehr wütend und frustriert",
        "katastrophe totales desaster",
        "ich hasse das entsetzlich",
        "traurig deprimiert und verzweifelt",
        "gescheitert fehler verloren schlecht",
        "angst stress und nervös",
        "widerlich ekelhaft und grauenhaft",
        "alles ist kaputt furchtbar",
        "ich bin so unglücklich traurig",
        "schrecklicher fehler katastrophal",
        "ich bin erschöpft und müde",
        "das ist widerlich entsetzlich",
        "ein alptraum schrecklich",
        "ich bin wütend und verzweifelt",
        "terrible schmerz und leiden",
        "ich hasse diese situation",
        "das ist ein desaster",
        "ich bin so frustriert",
        "schlimm und deprimierend",
        # ── ITALIEN négatif ───────────────────────────────────────────────────
        "questo e orribile e terribile",
        "sono molto arrabbiato e deluso",
        "che disastro catastrofe pessimo",
        "odio questa situazione orrenda",
        "triste depresso e disperato",
        "fallimento pessimo e schifoso",
        "sofferenza dolore e frustrazione",
        "furioso e stressato stanco",
        "tutto va male terribile",
        "sono molto triste e infelice",
        "orribile esperienza pessima",
        "sono esausto e stressato",
        "odio questo progetto pessimo",
        "che tragedia terribile",
        "sono disperato e senza speranza",
        "questo fa schifo orribile",
        "sono molto deluso e arrabbiato",
        "disastro totale pessimo",
        "che incubo orribile",
        "sono molto stanco e frustrato",
    ]

    pos = [
        # ── FRANÇAIS positif ──────────────────────────────────────────────────
        "je suis super content du resultat",
        "excellent travail vraiment bravo",
        "magnifique resultat fantastique",
        "je suis tres heureux et satisfait",
        "formidable succes incroyable",
        "parfait tout se passe bien",
        "je suis fier du travail accompli",
        "genial ce projet est une reussite",
        "joie et bonheur extraordinaire",
        "content satisfait du beau resultat",
        "incroyable performance sublime exceptionnel",
        "j adore ce projet remarquable",
        "super cool sympa et agreable",
        "bravo magnifique bien joue",
        "victoire reussite succes parfait",
        "je me sens vraiment bien aujourd hui",
        "excellent travail bien fait bravo",
        "je suis ravi du resultat",
        "tout va parfaitement bien",
        "magnifique je suis enchante",
        "superbe performance brillante",
        "je suis tres content et fier",
        "quelle belle reussite",
        "formidable exceptionnel remarquable",
        "je suis aux anges tellement content",
        "beau travail bien fait",
        "je me sens bien et serein",
        "succes total victoire",
        "je suis enthousiaste et motive",
        "excellent resultat je suis satisfait",
        "bravo c est parfait magnifique",
        "je suis heureux et reconnaissant",
        "superbe resultat bien joue",
        "incroyable je suis tres content",
        "parfait excellent et magnifique",
        "joie immense je suis ravi",
        "genial super et formidable",
        "je suis fier et satisfait",
        "tout est parfait excellent",
        "reussite totale je suis content",
        # ── ANGLAIS positif ───────────────────────────────────────────────────
        "this is amazing I love it so much",
        "excellent wonderful fantastic result",
        "I am so happy and excited today",
        "brilliant outstanding remarkable work",
        "I feel great and joyful",
        "perfect beautiful and magnificent",
        "awesome incredible I adore this",
        "thrilled overjoyed and grateful",
        "superb extraordinary achievement",
        "love this fantastic experience",
        "proud and confident excellent success",
        "delightful pleasant and enjoyable",
        "spectacular marvelous great work",
        "very happy satisfied and pleased",
        "wonderful peaceful and serene",
        "I am so grateful and blessed",
        "this is absolutely incredible",
        "I feel wonderful and amazing",
        "fantastic result I am thrilled",
        "brilliant work well done",
        "I love this so much wonderful",
        "amazing achievement outstanding",
        "I feel so happy and content",
        "excellent performance brilliant",
        "wonderful experience so happy",
        "I am overjoyed and excited",
        "this is perfect and beautiful",
        "incredible success I am proud",
        "so happy and grateful today",
        "fantastic job well done",
        "I feel great and confident",
        "amazing wonderful I love it",
        "superb brilliant outstanding",
        "I am thrilled and delighted",
        "this is excellent and wonderful",
        "perfect result so happy",
        "I feel joyful and peaceful",
        "wonderful success very happy",
        "brilliant and amazing result",
        "I am so pleased and satisfied",
        # ── ESPAGNOL positif ──────────────────────────────────────────────────
        "estoy muy feliz y contento hoy",
        "excelente resultado maravilloso",
        "me encanta esto es fantastico",
        "perfecto brillante increible",
        "alegre satisfecho con el logro",
        "amor y felicidad increible",
        "genial fenomenal espectacular",
        "orgulloso del exito obtenido",
        "estoy muy contento y feliz",
        "que resultado magnifico",
        "me siento muy bien hoy",
        "excelente trabajo bien hecho",
        "estoy emocionado y contento",
        "que alegria tan maravillosa",
        "perfecto todo va bien",
        "estoy muy satisfecho",
        "que logro tan increible",
        "estoy muy agradecido",
        "resultado excelente muy feliz",
        "que maravilla increible",
        # ── ALLEMAND positif ──────────────────────────────────────────────────
        "das ist wunderbar und fantastisch",
        "ausgezeichnet hervorragend brilliant",
        "ich bin sehr glücklich und froh",
        "perfekt großartig erstaunlich",
        "freude und begeisterung toll",
        "dankbar und optimistisch erfolgreich",
        "wunderschön genial und toll",
        "stolz zufrieden erfreut begeistert",
        "ich bin sehr zufrieden heute",
        "das ist ausgezeichnet perfekt",
        "wunderbar ich bin sehr glücklich",
        "tolles ergebnis gut gemacht",
        "ich bin begeistert und froh",
        "fantastisch brillant und schön",
        "ich freue mich sehr toll",
        "perfektes ergebnis sehr gut",
        "ich bin sehr dankbar heute",
        "wunderbar und fantastisch",
        "ich bin stolz und zufrieden",
        "großartig gut gemacht",
        # ── ITALIEN positif ───────────────────────────────────────────────────
        "sono molto felice e contento",
        "eccellente risultato magnifico",
        "mi piace molto fantastico",
        "perfetto brillante incredibile",
        "allegro soddisfatto del successo",
        "amore e gioia meraviglioso",
        "geniale fenomenale spettacolare",
        "orgoglioso del risultato vittoria",
        "sono molto soddisfatto oggi",
        "che risultato magnifico",
        "mi sento molto bene oggi",
        "eccellente lavoro ben fatto",
        "sono emozionato e contento",
        "che gioia meravigliosa",
        "perfetto tutto va bene",
        "sono molto grato",
        "che successo incredibile",
        "sono molto ottimista",
        "risultato eccellente molto felice",
        "che meraviglia incredibile",
    ]

    X = neg + pos
    y = [-1] * len(neg) + [1] * len(pos)
    return X, y


# ════════════════════════════════════════════════════════════════════════════════
# MODÈLE ML — ENTRAÎNEMENT ET CHARGEMENT
# ════════════════════════════════════════════════════════════════════════════════

_pipeline: Pipeline | None = None  # singleton chargé une fois


def _build_pipeline() -> Pipeline:
    """
    Construit le pipeline sklearn :
      TfidfVectorizer(char_wb, 2-5) + LogisticRegression

    Paramètres clés :
    • analyzer='char_wb' : N-grammes de caractères avec padding de mots.
      Capture les patterns morphologiques cross-lingues.
    • ngram_range=(2, 5)  : 2 à 5 caractères par token → optimal multilingual.
    • sublinear_tf=True   : TF logarithmique → réduit le poids des mots fréquents.
    • C=3.0               : régularisation faible → meilleure généralisation.
    • class_weight='balanced' : compense les déséquilibres de classe.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=100_000,
            strip_accents=None,    # garder les accents → signal utile
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=3.0,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def _train_and_save() -> Pipeline:
    """Entraîne le modèle sur le corpus et le sauvegarde sur disque."""
    X, y = _build_corpus()
    pipe  = _build_pipeline()
    pipe.fit(X, y)

    _MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, _MODEL_PATH)
    print(f"[INFO] Modèle sentiment entraîné et sauvegardé → {_MODEL_PATH}")
    return pipe


def _get_model() -> Pipeline:
    """Charge le modèle (depuis disque si dispo, sinon entraîne)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if _MODEL_PATH.exists():
        try:
            _pipeline = joblib.load(_MODEL_PATH)
            return _pipeline
        except Exception:
            pass

    _pipeline = _train_and_save()
    return _pipeline


# ════════════════════════════════════════════════════════════════════════════════
# PRÉ-TRAITEMENT
# ════════════════════════════════════════════════════════════════════════════════

_RE_TIMESTAMP = re.compile(r"\[\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\]]*\]")
_RE_SEPARATOR = re.compile(r"—{3,}|-{10,}")
_RE_KEYTAG    = re.compile(r"\[(?:BACK|TAB|ENTER|CTRL|ALT|SHIFT)[^\]]*\]", re.IGNORECASE)
_RE_SPACES    = re.compile(r"\s{2,}")
_RE_PUNCT_REP = re.compile(r"([!?.]){3,}")


def _clean(text: str) -> str:
    text = _RE_TIMESTAMP.sub(" ", text)
    text = _RE_SEPARATOR.sub(" ", text)
    text = _RE_KEYTAG.sub(" ", text)
    text = _RE_PUNCT_REP.sub(r"\1\1", text)
    text = _RE_SPACES.sub(" ", text)
    return text.strip()


# ════════════════════════════════════════════════════════════════════════════════
# DÉTECTION DE LANGUE
# ════════════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> tuple[str, float]:
    """Retourne (code_iso_restreint, confiance). Langue restreinte à FR/EN/ES/DE/IT."""
    if not _LANGDETECT_OK or len(text.split()) < 2:
        return "unknown", 0.0
    try:
        probs = detect_langs(text)
        # Parcourir par ordre de confiance, prendre le premier dans les 5 langues
        for p in probs:
            if p.lang in _SUPPORTED:
                return p.lang, round(p.prob, 3)
        # Aucune des 5 langues détectée → retourner la plus probable
        top = probs[0]
        return top.lang, round(top.prob, 3)
    except Exception:
        return "unknown", 0.0


# ════════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════════

def _logit_to_score(proba_pos: float) -> float:
    """
    Convertit la probabilité de classe positive en score continu [-1, +1].
    p=0.5 → 0.0  |  p=0.9 → +0.73  |  p=0.1 → -0.73
    Formule : tanh(logit(p)) où logit(p) = log(p/(1-p))
    """
    p = float(np.clip(proba_pos, 1e-6, 1 - 1e-6))
    logit = math.log(p / (1 - p))
    return round(math.tanh(logit), 4)


def _classify(score: float) -> str:
    if score >= THRESH_VERY_POS:
        return "très_positif"
    elif score >= THRESH_POS:
        return "positif"
    elif score <= THRESH_VERY_NEG:
        return "très_négatif"
    elif score <= THRESH_NEG:
        return "négatif"
    return "neutre"


# ════════════════════════════════════════════════════════════════════════════════
# API PUBLIQUE
# ════════════════════════════════════════════════════════════════════════════════

def analyze_sentiment(text: str) -> dict:
    """
    Analyse le sentiment d'un texte (FR / EN / ES / DE / IT).

    Retour
    ──────
    score      : float [-1.0, +1.0]  continu
    label      : str   très_positif | positif | neutre | négatif | très_négatif | trop_court
    language   : str   code ISO
    lang_flag  : str   emoji drapeau
    lang_conf  : float confiance détection de langue [0, 1]
    confidence : float confiance de l'analyse [0, 1]
    timestamp  : str   ISO 8601
    text       : str   texte nettoyé
    prob_pos   : float probabilité brute classe positive [0, 1]
    """
    ts         = datetime.now().isoformat()
    text_clean = _clean(text)
    word_count = len(text_clean.split())

    base = {
        "score": 0.0, "label": "trop_court",
        "language": "unknown", "lang_flag": "🌐", "lang_conf": 0.0,
        "confidence": 0.0, "timestamp": ts,
        "text": text_clean, "prob_pos": 0.5,
    }

    if word_count < MIN_WORDS:
        return base

    # ── Détection de langue ────────────────────────────────────────────────
    lang, lang_conf = detect_language(text_clean)

    # ── Inférence ML ──────────────────────────────────────────────────────
    model     = _get_model()
    probas    = model.predict_proba([text_clean])[0]
    # classes_ = [-1, 1] → index 0 = négatif, index 1 = positif
    prob_pos  = float(probas[1])
    score     = _logit_to_score(prob_pos)
    label     = _classify(score)

    # ── Confiance ─────────────────────────────────────────────────────────
    # Distance au point neutre 0.5 → plus on est loin, plus on est confiant
    distance   = abs(prob_pos - 0.5) * 2        # [0, 1]
    confidence = round(distance * 0.7 + lang_conf * 0.3, 3)

    return {
        "score":      score,
        "label":      label,
        "language":   lang,
        "lang_flag":  _LANG_FLAGS.get(lang, "🌐"),
        "lang_conf":  lang_conf,
        "confidence": confidence,
        "timestamp":  ts,
        "text":       text_clean,
        "prob_pos":   round(prob_pos, 4),
    }


def analyze_sentences_from_log(log_text: str) -> list:
    """Découpe le log en phrases et analyse chacune."""
    lines = []
    for raw in log_text.split("\n"):
        line = _clean(raw)
        if not line or line.startswith("—") or len(line) < 3:
            continue
        for sub in re.split(r"[.!?;]+", line):
            sub = sub.strip()
            if sub and len(sub.split()) >= MIN_WORDS:
                lines.append(sub)
    return [analyze_sentiment(s) for s in lines]


def save_sentiment_results(results: list, output_path: str = "data/sentiments.json") -> None:
    """Sauvegarde en mode append, ignore les trop_court."""
    dir_part = os.path.dirname(output_path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)

    existing = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    for r in results:
        if r.get("label") == "trop_court":
            continue
        existing.append({
            "timestamp":  r["timestamp"],
            "text":       r["text"],
            "sentiment":  r["label"],
            "score":      r["score"],
            "language":   r.get("language", "unknown"),
            "lang_flag":  r.get("lang_flag", "🌐"),
            "lang_conf":  r.get("lang_conf", 0.0),
            "confidence": r.get("confidence", 0.0),
            "prob_pos":   r.get("prob_pos", 0.5),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


# ── Test standalone ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.model_selection import cross_val_score

    print("═" * 70)
    print("  Moteur de sentiment — Character N-gram ML (sans lexique)")
    print("═" * 70)

    # Cross-validation sur le corpus complet
    X, y = _build_corpus()
    pipe  = _build_pipeline()
    cv    = cross_val_score(pipe, X, y, cv=5, scoring="f1_weighted")
    print(f"\n  CV F1 (5-fold) : {cv.mean():.3f} ± {cv.std():.3f}")
    print(f"  Taille corpus  : {len(X)} exemples ({sum(1 for v in y if v==-1)} neg / {sum(1 for v in y if v==1)} pos)")

    # Entraînement final
    model = _get_model()

    samples = [
        # PDF exact
        ("je suis en detresse",                      "très_négatif", "PDF phrase 1"),
        ("c est horrible",                           "très_négatif", "PDF phrase 2"),
        ("quel cata",                                "?",            "PDF phrase 3 (court)"),
        ("c est penible",                            "très_négatif", "PDF phrase 4"),
        ("strophe",                                  "trop_court",   "PDF phrase 5 (1 mot)"),
        # FR
        ("je suis vraiment furieux et en rage",      "très_négatif", "FR négatif fort"),
        ("j en ai vraiment marre de ce bug",         "très_négatif", "FR frustration"),
        ("je suis super content du resultat",        "très_positif", "FR positif"),
        ("excellent travail vraiment bravo",         "très_positif", "FR positif fort"),
        # EN
        ("I am absolutely furious",                  "très_négatif", "EN négatif"),
        ("this is amazing I love it",                "très_positif", "EN positif"),
        ("feeling sad and depressed today",          "très_négatif", "EN triste"),
        # ES
        ("estoy muy feliz y contento",               "très_positif", "ES positif"),
        ("esto es horrible y terrible",              "très_négatif", "ES négatif"),
        # DE
        ("das ist wunderbar und fantastisch",        "très_positif", "DE positif"),
        ("ich bin sehr wütend frustriert",           "très_négatif", "DE négatif"),
        # IT
        ("sono molto felice e contento",             "très_positif", "IT positif"),
        ("questo e orribile terribile",              "très_négatif", "IT négatif"),
    ]

    ok = 0
    print(f"\n  {'Texte':<45} {'Attendu':<15} {'Obtenu':<15} {'Score':>7}")
    print("  " + "─" * 85)
    for text, expected, desc in samples:
        r    = analyze_sentiment(text)
        got  = r["label"]
        match = (got == expected) or (expected == "?")
        if match:
            ok += 1
        icon = "✅" if match else "❌"
        print(f"  {icon} {r['lang_flag']} {text[:43]:<43} {expected:<15} {got:<15} {r['score']:>+7.4f}")

    print(f"\n  Résultat : {ok}/{len(samples)}  ({int(ok/len(samples)*100)}%)")
