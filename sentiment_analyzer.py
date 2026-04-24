"""
sentiment_analyzer.py  —  Moteur de sentiment MULTILINGUE
TP1 — Intelligence Artificielle & Cybersécurité

ARCHITECTURE v3 — "LangDetect + Lexique universel + Score hybride"
═══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────┐
  │  Texte keylogger (brut)                                     │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                   _clean()  ← suppression timestamps / balises
                         │
         ┌───────────────▼───────────────┐
         │  langdetect.detect_langs()    │   → langue + confiance
         │  (fr / en / es / de / it /   │
         │   pt / ar / zh / ja / nl …)  │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────────────────────────┐
         │  _score_lexicon(text, lang)                       │
         │  ┌────────────────────────────────────────────┐   │
         │  │ LEXIQUES PAR LANGUE  (intégrés, ~300 mots) │   │
         │  │  FR · EN · ES · DE · IT · PT · NL          │   │
         │  │  + UNIVERSEL (emoji, ponctuation, intensif) │   │
         │  └────────────────────────────────────────────┘   │
         │  → score brut [-4, +4]                            │
         └───────────────┬───────────────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  Normalisation tanh           │   → compound [-1, +1]
         │  compound = tanh(raw / 4)     │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  Classification 5 niveaux     │
         │  très_négatif / négatif /     │
         │  neutre / positif /           │
         │  très_positif                 │
         └───────────────────────────────┘

Langues supportées (détection + lexique) :
  fr · en · es · de · it · pt · nl
Langues avec détection seule (lexique universel) :
  ar · zh · ja · ko · ru · pl · et al.

Dépendances : langdetect (pip install langdetect)  — aucune autre
"""

import json
import math
import os
import re
import unicodedata
from datetime import datetime
from typing import Optional

try:
    from langdetect import detect_langs, DetectorFactory
    DetectorFactory.seed = 42          # reproductibilité
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False
    print("[AVERTISSEMENT] langdetect non installé : pip install langdetect")


# ══════════════════════════════════════════════════════════════════════════════
# LEXIQUES MULTILINGUES
# Chaque entrée : (mot, score)  score ∈ [-4, +4]
# Règle : |score| > 3 = très fort, 2-3 = fort, 1-2 = modéré, <1 = faible
# ══════════════════════════════════════════════════════════════════════════════

# ── Lexique universel (émojis, ponctuation répétée, intensificateurs latins) ─
_LEX_UNIVERSAL: dict[str, float] = {
    # Mots très négatifs/positifs présents dans plusieurs langues (filet de sécurité)
    "cata": -2.8, "disaster": -3.0, "super": 2.0, "ok": 0.8,
    # Émojis textuels
    ":)": 2.0, ":-)": 2.0, ":D": 2.5, "^_^": 2.0, "^^": 1.8,
    ":(": -2.0, ":-(": -2.0, ":/": -1.2, ":|": -0.8,
    "<3": 2.5, "</3": -2.5, ":P": 1.2, ";)": 1.5,
    "xd": 1.8, "XD": 1.8, "lol": 1.2, "lmao": 1.5,
    ":')": 1.5, ":'(": -2.0, "T_T": -2.0, ">:(": -2.5,
    # Intensificateurs romains
    "super": 1.8, "ultra": 1.6, "mega": 1.5, "hyper": 1.4, "extra": 1.3,
    "non": -0.5, "not": -0.5, "no": -0.4, "never": -0.8, "jamais": -0.7,
    "pas": -0.6, "ne": -0.3, "sans": -0.3, "rien": -0.5,
    # Ponctuations répétées (détectées par preprocessing)
    "!!!": 1.5, "???": -0.8, "...": -0.5,
}

# ── Français ──────────────────────────────────────────────────────────────────
_LEX_FR: dict[str, float] = {
    # Très négatifs
    "catastrophe": -3.5, "cata": -2.8, "desastre": -3.5, "désastre": -3.5,
    "cauchemar": -3.2, "horrible": -3.2, "atroce": -3.5, "abominable": -3.5,
    "infect": -3.0, "insupportable": -3.0, "inacceptable": -2.8,
    "epouvantable": -3.2, "épouvantable": -3.2, "effroyable": -3.2,
    "detestable": -3.0, "détestable": -3.0, "haïssable": -3.0,
    "horreur": -3.2, "terreur": -3.0, "panique": -2.8, "terreur": -3.0,
    # Négatifs forts
    "terrible": -2.8, "affreux": -2.8, "affreuse": -2.8,
    "nul": -2.5, "nulle": -2.5, "minable": -2.5, "mediocre": -2.3,
    "pourri": -2.8, "naze": -2.5, "lamentable": -2.8, "pitoyable": -2.8,
    "deteste": -3.0, "déteste": -3.0, "haïs": -3.5, "haine": -3.5,
    "hais": -3.5, "odieux": -2.8, "odieuse": -2.8,
    "detresse": -2.8, "détresse": -2.8, "desespoir": -3.2, "désespoir": -3.2,
    "desespere": -3.0, "désespéré": -3.0, "sombre": -2.0, "noir": -1.5,
    "deprime": -2.8, "déprimé": -2.8, "deprimee": -2.8, "déprimée": -2.8,
    "depression": -3.0, "dépression": -3.0, "melancolie": -2.5,
    "malheureux": -2.5, "malheureuse": -2.5, "triste": -2.5, "tristesse": -2.8,
    "pleure": -2.2, "pleurer": -2.2, "larmes": -2.0, "chagrin": -2.5,
    "souffre": -2.8, "souffrir": -2.8, "souffrance": -3.0, "douleur": -2.8,
    "douloureux": -2.5, "blesse": -2.5, "blessé": -2.5,
    # Colère
    "furieux": -3.0, "furieuse": -3.0, "rage": -3.0, "enrage": -3.0,
    "enragé": -3.0, "colere": -2.8, "colère": -2.8, "énervé": -2.5,
    "enerve": -2.5, "agace": -2.2, "agacé": -2.2, "irrite": -2.3,
    "irrité": -2.3, "exaspere": -2.8, "exaspéré": -2.8,
    "marre": -2.5, "ras-le-bol": -2.8, "bof": -1.0, "bouffe": -0.5,
    # Peur / Anxiété
    "peur": -2.0, "terrif": -3.0, "anxieux": -2.2, "anxieuse": -2.2,
    "angoisse": -2.8, "angoissé": -2.8, "inquiet": -2.0, "inquiete": -2.0,
    "stresse": -2.2, "stressé": -2.2, "stressée": -2.2, "stress": -1.8,
    "preoccupe": -1.8, "préoccupé": -1.8, "soucieux": -1.8,
    # Inconfort
    "penible": -2.5, "pénible": -2.5, "chiant": -2.5, "chiante": -2.5,
    "fatigue": -1.8, "fatigué": -2.0, "fatiguée": -2.0,
    "epuise": -2.5, "épuisé": -2.5, "epuisee": -2.5, "épuisée": -2.5,
    "difficile": -1.5, "complique": -1.5, "compliqué": -1.5,
    "impossible": -1.8, "bloque": -1.5, "bloqué": -1.5,
    "probleme": -1.8, "problème": -1.8, "souci": -1.5,
    "erreur": -1.8, "bug": -1.5, "panne": -2.0, "casse": -2.0, "cassé": -2.0,
    "raté": -2.5, "rate": -2.3, "echoue": -2.5, "échoué": -2.5,
    "echec": -2.8, "échec": -2.8, "loupe": -2.0, "loupé": -2.0,
    "perdu": -2.0, "perdre": -1.8, "mauvais": -2.0, "mauvaise": -2.0,
    "mal": -1.8, "horrible": -3.2, "pas bien": -1.5,
    # Négatifs modérés
    "ennuyeux": -1.5, "ennui": -1.5, "barbant": -1.5, "chiant": -2.0,
    "moche": -1.8, "laid": -1.5, "laide": -1.5, "ugly": -1.8,
    # Intensificateurs négatifs
    "vraiment": 1.3, "tres": 1.4, "très": 1.4, "tellement": 1.4,
    "trop": 1.3, "extremement": 1.6, "extrêmement": 1.6,
    "totalement": 1.3, "completement": 1.3, "complètement": 1.3,
    "absolument": 1.4, "franchement": 1.2, "vachement": 1.3,
    "profondement": 1.4, "profondément": 1.4, "terriblement": 1.5,
    # Positifs forts
    "excellent": 3.5, "parfait": 3.2, "parfaite": 3.2,
    "magnifique": 3.5, "fantastique": 3.2, "fabuleux": 3.0,
    "genial": 3.0, "génial": 3.0, "geniale": 3.0, "géniale": 3.0,
    "formidable": 3.0, "extraordinaire": 3.2, "incroyable": 2.8,
    "sublime": 3.2, "exceptionnel": 3.0, "remarquable": 2.8,
    "impressionnant": 2.8, "bluffant": 2.8,
    # Positifs modérés
    "bien": 1.8, "bon": 1.8, "bonne": 1.8, "super": 2.2, "top": 2.0,
    "cool": 1.8, "sympa": 1.8, "chouette": 1.8, "nickel": 2.0,
    "bravo": 2.5, "chapeau": 2.0, "felicitations": 3.0, "félicitations": 3.0,
    "merci": 1.5, "content": 2.3, "contente": 2.3,
    "heureux": 2.8, "heureuse": 2.8, "joie": 3.0, "bonheur": 3.2,
    "satisfait": 2.3, "satisfaite": 2.3, "agreable": 2.0, "agréable": 2.0,
    "adore": 3.0, "aime": 2.2, "aimer": 2.2, "amour": 2.8,
    "reussi": 2.5, "réussi": 2.5, "succes": 2.8, "succès": 2.8,
    "victoire": 2.8, "gagne": 2.5, "gagné": 2.5, "bravo": 2.5,
    "fier": 2.5, "fiere": 2.5, "fière": 2.5, "fierte": 2.5, "fierté": 2.5,
    "espoir": 2.2, "confiant": 2.0, "confiante": 2.0, "optimiste": 2.2,
    "beau": 2.0, "belle": 2.0, "joli": 1.8, "jolie": 1.8,
    "parfait": 3.0, "super": 2.2, "fantastique": 3.2,
    "calme": 1.5, "serein": 2.0, "sereine": 2.0, "paisible": 2.0,
    "mdr": 1.5, "lol": 1.5, "haha": 1.5, "hihi": 1.2,
}

# ── Anglais ───────────────────────────────────────────────────────────────────
_LEX_EN: dict[str, float] = {
    # Very negative
    "horrible": -3.2, "terrible": -3.0, "awful": -3.2, "dreadful": -3.0,
    "atrocious": -3.5, "abysmal": -3.2, "catastrophic": -3.5, "disaster": -3.2,
    "nightmare": -3.0, "disgusting": -3.0, "despise": -3.2, "hate": -3.5,
    "loathe": -3.5, "detest": -3.0, "abhor": -3.5,
    "devastating": -3.2, "unbearable": -3.0, "intolerable": -3.0,
    "worthless": -2.8, "useless": -2.5, "pathetic": -2.8, "miserable": -3.0,
    "hopeless": -3.0, "despair": -3.2, "depression": -3.0, "depressed": -2.8,
    "heartbroken": -3.2, "devastated": -3.2, "destroyed": -2.8,
    "furious": -3.0, "rage": -3.0, "outraged": -3.0, "livid": -3.0,
    "angry": -2.5, "annoyed": -2.0, "irritated": -2.2, "frustrated": -2.5,
    "disgusted": -2.8, "appalled": -2.8,
    "scared": -2.2, "terrified": -3.0, "anxious": -2.2, "stressed": -2.2,
    "worried": -2.0, "panicked": -2.8, "nervous": -1.8,
    "sad": -2.5, "unhappy": -2.5, "upset": -2.3, "cry": -2.0,
    "crying": -2.2, "tears": -2.0, "grief": -2.8, "sorrow": -2.8,
    "pain": -2.5, "suffering": -3.0, "hurt": -2.3, "broken": -2.5,
    "failed": -2.5, "failure": -2.8, "lost": -2.0, "losing": -2.0,
    "bad": -2.0, "worst": -3.2, "poor": -1.8, "wrong": -1.8,
    "boring": -1.5, "dull": -1.5, "stupid": -2.5, "idiot": -2.8,
    # Intensifiers
    "very": 1.4, "really": 1.3, "extremely": 1.6, "absolutely": 1.5,
    "totally": 1.3, "completely": 1.4, "utterly": 1.5, "incredibly": 1.4,
    "deeply": 1.4, "profoundly": 1.5, "terribly": 1.5, "awfully": 1.4,
    # Negations
    "not": -0.6, "no": -0.5, "never": -0.8, "none": -0.5,
    "without": -0.3, "neither": -0.4, "nor": -0.4, "barely": -0.5,
    # Very positive
    "excellent": 3.5, "outstanding": 3.5, "exceptional": 3.5,
    "amazing": 3.2, "fantastic": 3.2, "fabulous": 3.0, "wonderful": 3.2,
    "brilliant": 3.0, "magnificent": 3.2, "spectacular": 3.2,
    "superb": 3.0, "extraordinary": 3.2, "incredible": 2.8, "awesome": 3.0,
    "perfect": 3.2, "flawless": 3.0, "marvelous": 3.0,
    # Positive moderate
    "good": 2.0, "great": 2.5, "nice": 1.8, "fine": 1.5, "well": 1.5,
    "happy": 2.8, "joy": 3.0, "joyful": 3.0, "glad": 2.3, "pleased": 2.2,
    "excited": 2.5, "thrilled": 2.8, "delighted": 2.8, "overjoyed": 3.2,
    "love": 3.0, "adore": 3.0, "like": 1.8, "enjoy": 2.2,
    "satisfied": 2.2, "content": 2.0, "comfortable": 1.8,
    "proud": 2.5, "confident": 2.0, "hopeful": 2.2, "optimistic": 2.2,
    "grateful": 2.5, "thankful": 2.3, "blessed": 2.5,
    "beautiful": 2.5, "lovely": 2.3, "gorgeous": 2.5, "pretty": 1.8,
    "fun": 2.0, "funny": 1.8, "hilarious": 2.3, "laugh": 1.8,
    "success": 2.8, "won": 2.5, "win": 2.5, "achieved": 2.3, "accomplished": 2.5,
    "calm": 1.5, "peaceful": 2.0, "relaxed": 1.8, "serene": 2.0,
    "thanks": 1.5, "thank": 1.5, "appreciate": 2.0,
    "lol": 1.5, "haha": 1.5, "hehe": 1.2,
}

# ── Espagnol ──────────────────────────────────────────────────────────────────
_LEX_ES: dict[str, float] = {
    # Muy negativo
    "horrible": -3.2, "terrible": -3.0, "espantoso": -3.2, "desastre": -3.5,
    "catastrofe": -3.5, "catástrofe": -3.5, "odio": -3.5, "detesto": -3.0,
    "asco": -3.0, "asqueroso": -3.2, "espantosa": -3.2, "pesimo": -3.0,
    "pésimo": -3.0, "maldito": -2.8, "fatal": -2.8,
    "furioso": -3.0, "furiosa": -3.0, "enojado": -2.5, "enojada": -2.5,
    "enfadado": -2.5, "irritado": -2.3, "frustrado": -2.5, "frustrada": -2.5,
    "triste": -2.5, "tristeza": -2.8, "llorar": -2.2, "lloro": -2.2,
    "sufrimiento": -3.0, "sufrir": -2.8, "dolor": -2.8,
    "miedo": -2.2, "aterrado": -3.0, "ansioso": -2.2, "estresado": -2.2,
    "mal": -2.0, "peor": -2.5, "malo": -2.0, "mala": -2.0,
    "aburrido": -1.5, "pesado": -1.8, "cansado": -1.8, "agotado": -2.5,
    "fracaso": -2.8, "perder": -2.0, "fallé": -2.5, "fallo": -2.3,
    # Intensificadores
    "muy": 1.4, "mucho": 1.3, "extremadamente": 1.6, "totalmente": 1.4,
    "absolutamente": 1.5, "realmente": 1.3, "increíblemente": 1.5,
    "no": -0.5, "nunca": -0.8, "sin": -0.3, "jamás": -0.8,
    # Muy positivo
    "excelente": 3.5, "magnifico": 3.2, "magnifico": 3.2, "fantástico": 3.2,
    "maravilloso": 3.2, "increíble": 2.8, "espectacular": 3.0,
    "perfecto": 3.2, "brillante": 3.0, "genial": 3.0, "fenomenal": 3.0,
    # Positivo moderado
    "feliz": 2.8, "alegre": 2.5, "contento": 2.3, "contenta": 2.3,
    "bien": 1.8, "bueno": 1.8, "buena": 1.8, "bonito": 2.0, "bonita": 2.0,
    "amor": 3.0, "quiero": 2.2, "encanta": 2.8, "adoro": 3.0,
    "gracias": 1.8, "agradecido": 2.5, "orgulloso": 2.5,
    "exito": 2.8, "éxito": 2.8, "logro": 2.5, "gané": 2.5,
    "guapo": 2.0, "hermoso": 2.5, "hermosa": 2.5, "precioso": 2.5,
}

# ── Allemand ──────────────────────────────────────────────────────────────────
_LEX_DE: dict[str, float] = {
    # Sehr negativ
    "furchtbar": -3.2, "schrecklich": -3.2, "horrible": -3.0, "katastrophe": -3.5,
    "desaster": -3.5, "schlimm": -2.8, "schlecht": -2.3, "übel": -2.5,
    "widerlich": -3.0, "ekelhaft": -3.0, "grauenhaft": -3.2, "entsetzlich": -3.2,
    "hassenswert": -3.0, "hasse": -3.5, "hassen": -3.5, "verabscheue": -3.5,
    "wütend": -3.0, "wut": -3.0, "zornig": -2.8, "verärgert": -2.5,
    "frustriert": -2.5, "gereizt": -2.3, "genervt": -2.5,
    "traurig": -2.5, "traurigkeit": -2.8, "deprimiert": -2.8, "verzweifelt": -3.0,
    "leiden": -2.8, "schmerz": -2.8, "angst": -2.5, "stress": -2.0,
    "gestresst": -2.3, "besorgt": -2.0, "nervös": -2.0,
    "schlecht": -2.3, "schlechter": -2.5, "schlechteste": -3.0,
    "langweilig": -1.5, "müde": -1.8, "erschöpft": -2.5,
    "gescheitert": -2.8, "fehler": -1.8, "verloren": -2.0,
    # Intensivierer
    "sehr": 1.4, "wirklich": 1.3, "extrem": 1.6, "total": 1.4,
    "absolut": 1.5, "wirklich": 1.3, "unglaublich": 1.5,
    "nicht": -0.6, "kein": -0.5, "nie": -0.8, "niemals": -0.9,
    # Sehr positiv
    "ausgezeichnet": 3.5, "hervorragend": 3.5, "wunderbar": 3.2,
    "fantastisch": 3.2, "großartig": 3.0, "erstaunlich": 2.8,
    "brilliant": 3.0, "perfekt": 3.2, "wunderschön": 3.2, "genial": 3.0,
    # Positiv moderat
    "gut": 1.8, "schön": 2.2, "toll": 2.5, "prima": 2.2, "super": 2.2,
    "glücklich": 2.8, "froh": 2.5, "zufrieden": 2.3, "erfreut": 2.5,
    "freude": 3.0, "begeistert": 2.8, "liebe": 3.0, "mag": 2.0,
    "danke": 1.5, "dankbar": 2.5, "stolz": 2.5, "optimistisch": 2.2,
    "erfolgreich": 2.8, "gewonnen": 2.5, "gelungen": 2.5, "bravo": 2.5,
}

# ── Italien ───────────────────────────────────────────────────────────────────
_LEX_IT: dict[str, float] = {
    # Molto negativo
    "orribile": -3.2, "terribile": -3.0, "spaventoso": -3.0, "disastro": -3.5,
    "catastrofe": -3.5, "odio": -3.5, "detesto": -3.0, "disgustoso": -3.0,
    "pessimo": -3.0, "pessima": -3.0, "schifoso": -3.0, "orrendo": -3.2,
    "arrabbiato": -2.8, "arrabbiata": -2.8, "furioso": -3.0, "furiosa": -3.0,
    "frustrato": -2.5, "frustrata": -2.5, "deluso": -2.5, "delusa": -2.5,
    "triste": -2.5, "tristezza": -2.8, "piango": -2.2, "piangere": -2.2,
    "sofferenza": -3.0, "dolore": -2.8, "paura": -2.2,
    "stress": -2.0, "stressato": -2.2, "stanco": -1.8, "esausto": -2.5,
    "male": -2.0, "peggio": -2.5, "cattivo": -2.0, "brutto": -2.0,
    "fallito": -2.8, "fallimento": -2.8, "perso": -2.0,
    # Intensificatori
    "molto": 1.4, "davvero": 1.3, "estremamente": 1.6, "assolutamente": 1.5,
    "non": -0.6, "mai": -0.8, "senza": -0.3,
    # Molto positivo
    "eccellente": 3.5, "magnifico": 3.2, "fantastico": 3.2, "meraviglioso": 3.2,
    "straordinario": 3.2, "incredibile": 2.8, "spettacolare": 3.0,
    "perfetto": 3.2, "brillante": 3.0, "geniale": 3.0, "fenomenale": 3.0,
    # Positivo moderato
    "felice": 2.8, "contento": 2.3, "contenta": 2.3, "allegro": 2.5,
    "bene": 1.8, "buono": 1.8, "buona": 1.8, "bello": 2.2, "bella": 2.2,
    "amore": 3.0, "adoro": 3.0, "amo": 2.8, "grazie": 1.8,
    "successo": 2.8, "vittoria": 2.8, "orgoglioso": 2.5,
}

# ── Portugais ─────────────────────────────────────────────────────────────────
_LEX_PT: dict[str, float] = {
    # Muito negativo
    "horrível": -3.2, "terrível": -3.0, "desastre": -3.5, "catástrofe": -3.5,
    "odeio": -3.5, "detesto": -3.0, "nojento": -3.0, "péssimo": -3.0,
    "furioso": -3.0, "furiosa": -3.0, "irritado": -2.5, "frustrado": -2.5,
    "triste": -2.5, "tristeza": -2.8, "chorar": -2.2, "sofrimento": -3.0,
    "medo": -2.2, "ansioso": -2.2, "estressado": -2.2, "cansado": -1.8,
    "mau": -2.0, "ruim": -2.0, "pior": -2.5, "fracasso": -2.8,
    # Muito positivo
    "excelente": 3.5, "maravilhoso": 3.2, "fantástico": 3.2, "incrível": 2.8,
    "perfeito": 3.2, "brilhante": 3.0, "genial": 3.0,
    # Positivo moderado
    "feliz": 2.8, "alegre": 2.5, "contente": 2.3, "bem": 1.8, "bom": 1.8,
    "amor": 3.0, "adoro": 3.0, "obrigado": 1.8, "obrigada": 1.8,
    "sucesso": 2.8, "orgulhoso": 2.5,
    "muito": 1.4, "realmente": 1.3, "extremamente": 1.6,
    "não": -0.6, "nunca": -0.8,
}

# ── Néerlandais ───────────────────────────────────────────────────────────────
_LEX_NL: dict[str, float] = {
    "verschrikkelijk": -3.2, "afschuwelijk": -3.2, "rampzalig": -3.5,
    "haat": -3.5, "verafschuw": -3.5, "walgelijk": -3.0,
    "boos": -2.5, "woedend": -3.0, "gefrustreerd": -2.5,
    "verdrietig": -2.5, "depressief": -2.8, "hopeloos": -3.0,
    "slecht": -2.3, "erg": -2.0, "vreselijk": -3.0,
    "uitstekend": 3.5, "prachtig": 3.2, "fantastisch": 3.2,
    "geweldig": 3.0, "schitterend": 3.2, "perfect": 3.2,
    "gelukkig": 2.8, "blij": 2.5, "tevreden": 2.3, "goed": 1.8,
    "liefde": 3.0, "dankbaar": 2.5, "trots": 2.5,
    "heel": 1.4, "echt": 1.3, "absoluut": 1.5,
    "niet": -0.6, "nooit": -0.8,
}

# ── Map langue → lexique ──────────────────────────────────────────────────────
_LANG_TO_LEX: dict[str, dict] = {
    "fr": _LEX_FR,
    "en": _LEX_EN,
    "es": _LEX_ES,
    "de": _LEX_DE,
    "it": _LEX_IT,
    "pt": _LEX_PT,
    "nl": _LEX_NL,
}

# Drapeaux pour l'affichage
_LANG_FLAGS: dict[str, str] = {
    "fr": "🇫🇷", "en": "🇬🇧", "es": "🇪🇸", "de": "🇩🇪",
    "it": "🇮🇹", "pt": "🇵🇹", "nl": "🇳🇱", "ar": "🇸🇦",
    "zh": "🇨🇳", "ja": "🇯🇵", "ko": "🇰🇷", "ru": "🇷🇺",
    "pl": "🇵🇱", "sv": "🇸🇪", "da": "🇩🇰", "fi": "🇫🇮",
    "tr": "🇹🇷", "ro": "🇷🇴", "cs": "🇨🇿", "hu": "🇭🇺",
    "unknown": "🌐",
}

# ══════════════════════════════════════════════════════════════════════════════
# SEUILS
# ══════════════════════════════════════════════════════════════════════════════
THRESH_VERY_POS  =  0.50
THRESH_POS       =  0.10
THRESH_NEG       = -0.10
THRESH_VERY_NEG  = -0.50
MIN_WORDS        =  2    # Seuil bas pour capturer les textes courts multilingues


# ══════════════════════════════════════════════════════════════════════════════
# PRÉ-TRAITEMENT
# ══════════════════════════════════════════════════════════════════════════════
_RE_TIMESTAMP = re.compile(r'\[\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\]]*\]')
_RE_SEPARATOR = re.compile(r'—{3,}|-{10,}')
_RE_KEYTAG    = re.compile(r'\[(?:BACK|TAB|ENTER|CTRL|ALT|SHIFT)[^\]]*\]', re.IGNORECASE)
_RE_SPACES    = re.compile(r'\s{2,}')
_RE_PUNCT_REP = re.compile(r'([!?.]){2,}')  # !!! → un seul marqueur fort


def _clean(text: str) -> str:
    """Nettoie le texte keylogger avant analyse."""
    text = _RE_TIMESTAMP.sub(' ', text)
    text = _RE_SEPARATOR.sub(' ', text)
    text = _RE_KEYTAG.sub(' ', text)
    text = _RE_PUNCT_REP.sub(r'\1\1', text)   # !!! → !!
    text = _RE_SPACES.sub(' ', text)
    return text.strip()


def _normalize_word(w: str) -> str:
    """Minuscule + suppression des accents pour lookup lexique."""
    w = w.lower()
    nfkd = unicodedata.normalize('NFD', w)
    return ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')


# ══════════════════════════════════════════════════════════════════════════════
# DÉTECTION DE LANGUE
# ══════════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> tuple[str, float]:
    """
    Détecte la langue du texte avec langdetect.
    Retourne (code_iso, confiance) ou ('unknown', 0.0).
    """
    if not _LANGDETECT_OK or len(text.split()) < 2:
        return "unknown", 0.0
    try:
        probs = detect_langs(text)
        top   = probs[0]
        return top.lang, round(top.prob, 3)
    except Exception:
        return "unknown", 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MOTEUR DE SCORING LEXICAL
# ══════════════════════════════════════════════════════════════════════════════

def _score_lexicon(text: str, lang: str) -> tuple[float, int]:
    """
    Calcule le score de sentiment brut via les lexiques.

    Algorithme :
    1. Tokenisation sur espaces et ponctuation
    2. Pour chaque token : lookup dans lexique universel + lexique de la langue
    3. Gestion des intensificateurs/négations sur le token suivant
    4. Agrégation linéaire

    Retourne (score_brut, nb_mots_significatifs)
    """
    lang_lex = _LANG_TO_LEX.get(lang, {})

    # Tokenisation
    tokens_raw = re.findall(r"[\w']+|[!?.:,;]+", text.lower())
    tokens     = tokens_raw[:]

    raw_score  = 0.0
    sig_words  = 0
    booster    = 1.0   # multiplicateur du prochain mot sémantique
    negate     = False  # prochaine valeur inversée

    _NEGATION_WORDS = {"not","no","never","ne","pas","jamais","rien","kein","nicht",
                       "no","non","nein","nunca","nao","niet","nooit","mai","nessun"}
    _BOOST_WORDS    = {"very","really","extremely","absolutely","totally","very","really",
                       "très","tres","vraiment","vraiment","tellement","tellement","trop",
                       "sehr","wirklich","extrem","molto","davvero","absolutamente",
                       "extremamente","heel","echt","super","hyper","mega","ultra"}

    for i, token in enumerate(tokens):
        norm = _normalize_word(token)

        # Vérif négation / intensificateur
        if norm in _NEGATION_WORDS:
            negate = True
            continue
        if norm in _BOOST_WORDS:
            booster *= 1.35
            continue

        # Lookup
        score = None
        # 1. Lexique langue spécifique (prioritaire)
        if token in lang_lex:
            score = lang_lex[token]
        elif norm in lang_lex:
            score = lang_lex[norm]
        # 2. Lexique universel
        elif token in _LEX_UNIVERSAL:
            score = _LEX_UNIVERSAL[token]
        elif norm in _LEX_UNIVERSAL:
            score = _LEX_UNIVERSAL[norm]

        if score is not None:
            effective = score * booster
            if negate:
                effective = -effective * 0.6   # négation partielle, pas totale
                negate = False
            raw_score += effective
            sig_words += 1
            booster = 1.0  # reset booster après usage

        # Ponctuation répétée → bonus d'intensité
        if re.match(r'^[!]+$', token):
            raw_score += 0.3 * len(token)  # !! = +0.6, !!! = +0.9

    return raw_score, sig_words


def _normalize_score(raw: float, n_words: int) -> float:
    """
    Normalise le score brut en [-1, +1] via tanh.
    La normalisation par n_words évite que les textes longs dominent.
    """
    if n_words == 0:
        return 0.0
    # Normaliser par la racine du nombre de mots (diminishing returns)
    normalized = raw / max(math.sqrt(max(n_words, 1)), 1)
    # tanh compresse dans [-1, +1]
    compound = math.tanh(normalized / 2.5)
    return round(compound, 4)


def _classify(compound: float) -> str:
    """Classe le score en 5 niveaux."""
    if compound >= THRESH_VERY_POS:
        return "très_positif"
    elif compound >= THRESH_POS:
        return "positif"
    elif compound <= THRESH_VERY_NEG:
        return "très_négatif"
    elif compound <= THRESH_NEG:
        return "négatif"
    return "neutre"


# ══════════════════════════════════════════════════════════════════════════════
# API PUBLIQUE
# ══════════════════════════════════════════════════════════════════════════════

def analyze_sentiment(text: str) -> dict:
    """
    Analyse le sentiment d'un texte dans n'importe quelle langue.

    Retour
    ------
    dict :
        score      : float [-1.0, +1.0]
        label      : str   très_positif | positif | neutre | négatif | très_négatif | trop_court
        language   : str   code ISO (fr, en, es, de, it, pt, nl, unknown…)
        lang_flag  : str   emoji drapeau
        lang_conf  : float confiance de la détection [0.0, 1.0]
        confidence : float confiance de l'analyse sentiment [0.0, 1.0]
        timestamp  : str   ISO 8601
        text       : str   texte nettoyé
        sig_words  : int   mots significatifs trouvés dans le lexique
    """
    ts         = datetime.now().isoformat()
    text_clean = _clean(text)
    word_count = len(text_clean.split())

    base = {
        "score": 0.0, "label": "trop_court",
        "language": "unknown", "lang_flag": "🌐", "lang_conf": 0.0,
        "confidence": 0.0, "timestamp": ts,
        "text": text_clean, "sig_words": 0,
    }

    if word_count < MIN_WORDS:
        return base

    # ── Détection de langue ────────────────────────────────────────────────
    lang, lang_conf = detect_language(text_clean)

    # Pour les textes courts (<6 mots) avec faible confiance, tester aussi FR et EN
    # car langdetect confond souvent le français court avec NL/CA/IT
    if word_count < 7 and lang_conf < 0.90 and lang not in ("fr", "en"):
        from sentiment_analyzer import _score_lexicon as _sl
        fr_raw, fr_sw = _score_lexicon(text_clean, "fr")
        en_raw, en_sw = _score_lexicon(text_clean, "en")
        if fr_sw > 0 or en_sw > 0:
            lang = "fr" if fr_sw >= en_sw else "en"
            lang_conf = 0.75  # confiance réduite

    # ── Scoring lexical ────────────────────────────────────────────────────
    raw_score, sig_words = _score_lexicon(text_clean, lang)

    # Fallback : si aucun mot significatif trouvé (langue non reconnue ou courte),
    # scanner TOUS les lexiques pour ne rien manquer
    if sig_words == 0 and lang not in _LANG_TO_LEX:
        for fallback_lang in ("fr", "en", "es", "de", "it", "pt", "nl"):
            rs, sw = _score_lexicon(text_clean, fallback_lang)
            if sw > sig_words:
                raw_score, sig_words = rs, sw
                lang = fallback_lang  # adopter la langue du meilleur match

    compound   = _normalize_score(raw_score, max(sig_words, 1))
    label      = _classify(compound)

    # ── Confiance : mots significatifs / total × confiance langue ─────────
    lex_coverage = min(sig_words / max(word_count, 1), 1.0)
    confidence   = round(
        (abs(compound) * 0.5 + lex_coverage * 0.3 + lang_conf * 0.2), 3
    )

    return {
        "score":      compound,
        "label":      label,
        "language":   lang,
        "lang_flag":  _LANG_FLAGS.get(lang, "🌐"),
        "lang_conf":  lang_conf,
        "confidence": confidence,
        "timestamp":  ts,
        "text":       text_clean,
        "sig_words":  sig_words,
    }


def analyze_sentences_from_log(log_text: str) -> list:
    """
    Découpe le log en phrases et analyse chacune.
    Séparateurs : saut de ligne, point, !, ?
    """
    lines = []
    for raw in log_text.split("\n"):
        line = _clean(raw)
        if not line or line.startswith("—") or len(line) < 3:
            continue
        for sub in re.split(r'[.!?;]+', line):
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
            "sig_words":  r.get("sig_words", 0),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


# ── Test standalone ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        # Français
        ("je suis vraiment en detresse", "fr"),
        ("c est horrible et penible", "fr"),
        ("je suis super content du resultat", "fr"),
        ("excellent travail vraiment bravo", "fr"),
        ("quel cata horrible desastre", "fr"),
        ("j en ai vraiment marre de ce bug", "fr"),
        # Anglais
        ("I am absolutely furious right now", "en"),
        ("This is amazing I love it so much", "en"),
        ("feeling very sad and depressed today", "en"),
        ("everything went perfectly well today", "en"),
        # Espagnol
        ("estoy muy feliz y contento hoy", "es"),
        ("esto es horrible y terrible", "es"),
        # Allemand
        ("das ist wunderbar und fantastisch", "de"),
        ("ich bin sehr wütend und frustriert", "de"),
        # Italien
        ("sono molto felice e contento", "it"),
        ("questo è orribile e terribile", "it"),
    ]

    print(f"\n{'Texte':<45} {'Lang':<5} {'Label':<15} {'Score':>7}  {'Conf'}")
    print("─" * 85)
    for text, expected_lang in samples:
        r = analyze_sentiment(text)
        ok = "✅" if r["language"] == expected_lang else "⚠️ "
        print(f"{ok} {r['lang_flag']} {text[:41]:<41} {r['language']:<5} "
              f"{r['label']:<15} {r['score']:>+7.4f}  {r['confidence']:.3f}")
