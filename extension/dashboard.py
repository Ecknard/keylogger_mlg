"""
extension/dashboard.py — Dashboard de supervision temps réel MULTILINGUE
TP1 — Intelligence Artificielle & Cybersécurité — v3

NOUVEAUTÉS v3 :
    🌐 Moteur de sentiment multilingue (FR/EN/ES/DE/IT/PT/NL + détection auto)
    🚀 Rafraîchissement Streamlit via st.rerun() avec compteur visible
    📊 Graphiques enrichis : carte des langues, confiance, radar
    🔴 Zéro cache — lecture directe fichiers JSON à chaque cycle
    ⚡ Indicateur de fraîcheur + avertissement keylogger inactif
    🏷️  Drapeaux emoji par phrase dans la table des sentiments
"""

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

# ── Config Streamlit ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Keylogger · Sentiment ML Multilingue",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #070b14;
    color: #c9d1d9;
}
.main { background-color: #070b14; }
.block-container { padding: 1.2rem 2rem; max-width: 1500px; }

/* ── Header ── */
.dash-header {
    background: linear-gradient(135deg, #08111f 0%, #0d1f3c 50%, #08111f 100%);
    border: 1px solid #162235;
    border-radius: 14px;
    padding: 24px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #00d4ff, #0066ff, #7b2fff, #ff6b35, #00d4ff);
    background-size: 300% 100%;
    animation: flow 4s linear infinite;
}
@keyframes flow { 0%{background-position:0 0} 100%{background-position:300% 0} }
.dash-header h1 { font-size:1.7em; font-weight:800; color:#e6edf3; margin:0 0 4px; }
.dash-header p  { color:#8b949e; font-size:0.85em; margin:0; font-family:'JetBrains Mono',monospace; }

/* ── KPI Grid ── */
.kpi-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-bottom:20px; }
.kpi-card {
    background:#0d1117; border:1px solid #1e2730; border-radius:10px;
    padding:18px 20px; position:relative; overflow:hidden; transition:border-color .2s;
}
.kpi-card:hover { border-color:#388bfd; }
.kpi-card::after {
    content:''; position:absolute; bottom:0; left:0; right:0; height:3px; border-radius:0 0 10px 10px;
}
.kpi-card.blue::after   { background:#388bfd; }
.kpi-card.green::after  { background:#3fb950; }
.kpi-card.red::after    { background:#f85149; }
.kpi-card.yellow::after { background:#d29922; }
.kpi-card.purple::after { background:#7b2fff; }
.kpi-card .kpi-value { font-size:2em; font-weight:800; font-family:'JetBrains Mono',monospace; color:#e6edf3; line-height:1; }
.kpi-card .kpi-label { font-size:0.73em; color:#8b949e; margin-top:5px; text-transform:uppercase; letter-spacing:.08em; }
.kpi-card .kpi-sub   { font-size:0.72em; color:#484f58; margin-top:6px; font-family:'JetBrains Mono',monospace; }
.kpi-card .kpi-icon  { position:absolute; right:16px; top:50%; transform:translateY(-50%); font-size:1.6em; opacity:.12; }

/* ── Section title ── */
.sec { font-family:'Syne',sans-serif; font-weight:700; font-size:.9em; color:#8b949e;
       text-transform:uppercase; letter-spacing:.12em; border-left:3px solid #388bfd;
       padding-left:10px; margin-bottom:12px; }

/* ── Status bar ── */
.statusbar {
    background:#010409; border:1px solid #1e2730; border-radius:8px;
    padding:9px 16px; margin-bottom:18px;
    display:flex; justify-content:space-between; align-items:center;
    font-family:'JetBrains Mono',monospace; font-size:.75em;
}
.live   { color:#3fb950; }
.live::before   { content:'● '; animation:blink 1.2s ease-in-out infinite; }
.stale  { color:#d29922; font-weight:700; }
.stale::before  { content:'⚠ '; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

/* ── Sentiment rows ── */
.sent-row {
    background:#0d1117; border:1px solid #1e2730; border-radius:8px;
    padding:10px 14px; margin-bottom:6px;
}
.sent-row .sent-text { font-family:'JetBrains Mono',monospace; font-size:.8em; color:#e6edf3; }
.sent-row .sent-meta { font-size:.7em; color:#484f58; margin-top:4px; }

/* ── Lang badge ── */
.lang-badge {
    display:inline-block; padding:2px 8px; border-radius:12px;
    font-size:.7em; font-weight:600; font-family:'JetBrains Mono',monospace;
    background:rgba(56,139,253,.12); color:#388bfd; border:1px solid rgba(56,139,253,.3);
    margin-left:6px;
}

/* ── Alert badge ── */
.badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:.73em; font-weight:600; font-family:'JetBrains Mono',monospace;
}
.b-pos   { background:rgba(63,185,80,.12);  color:#3fb950; border:1px solid #3fb950; }
.b-neg   { background:rgba(248,81,73,.12);  color:#f85149; border:1px solid #f85149; }
.b-neu   { background:rgba(139,148,158,.1); color:#8b949e; border:1px solid #484f58; }
.b-warn  { background:rgba(210,153,34,.12); color:#d29922; border:1px solid #d29922; }
.b-crit  { background:rgba(248,81,73,.15);  color:#f85149; border:1px solid #f85149; }
.b-info  { background:rgba(56,139,253,.12); color:#388bfd; border:1px solid #388bfd; }

/* ── Detection row ── */
.det-row {
    background:#0d1117; border:1px solid #1e2730; border-radius:8px;
    padding:10px 14px; margin-bottom:6px;
    display:flex; justify-content:space-between; align-items:center;
}
.det-row .det-time { font-size:.72em; color:#484f58; }

/* ── Log viewer ── */
.log-box {
    background:#010409; border:1px solid #1e2730; border-radius:8px;
    padding:14px; height:300px; overflow-y:auto;
    font-family:'JetBrains Mono',monospace; font-size:.78em; line-height:1.8;
}

/* ── Progress bar ── */
.pbar-wrap { background:#1e2730; border-radius:4px; height:6px; margin-top:6px; }
.pbar-fill { height:6px; border-radius:4px; transition:width .3s; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#010409; }
::-webkit-scrollbar-thumb { background:#1e2730; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#388bfd; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:#0d1117; border-right:1px solid #1e2730; }

/* ── Streamlit overrides ── */
.stSelectbox > div > div { background:#0d1117; border-color:#1e2730; color:#e6edf3; }
div[data-testid="metric-container"] { background:#0d1117; border:1px solid #1e2730; border-radius:8px; padding:12px; }
button[kind="primary"] { background:#388bfd; border:none; border-radius:6px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES — ZÉRO CACHE
# ════════════════════════════════════════════════════════════════════════════════

def _read_json(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _read_log_tail(path: Path, n: int = 80) -> list:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()[-n:]
    except Exception:
        return []


def _file_age(path: Path) -> float:
    """Âge du fichier en secondes. +inf si absent."""
    if not path.exists():
        return float("inf")
    return time.time() - path.stat().st_mtime


def load_all() -> dict:
    """Lecture directe sans aucun cache."""
    return {
        "sentiments": _read_json(DATA / "sentiments.json"),
        "alerts":     _read_json(DATA / "alerts.json"),
        "detections": _read_json(DATA / "detections.json"),
        "metadata":   _read_json(DATA / "metadata.json"),
        "log_lines":  _read_log_tail(DATA / "log.txt"),
        "ts":         datetime.now(),
        "age_log":    _file_age(DATA / "log.txt"),
        "age_sent":   _file_age(DATA / "sentiments.json"),
    }


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _is_recent(ts_str: str, minutes: int = 60) -> bool:
    try:
        return datetime.now() - datetime.fromisoformat(ts_str) < timedelta(minutes=minutes)
    except Exception:
        return False


def _valid_sents(sents: list) -> list:
    """Filtre les entrées trop_court et erreur_librairie."""
    skip = {"trop_court", "erreur_librairie"}
    return [s for s in sents if s.get("sentiment", s.get("label", "")) not in skip]


# Couleurs par label
_LABEL_COLOR = {
    "très_positif":  "#00ff88",
    "positif":       "#3fb950",
    "neutre":        "#8b949e",
    "négatif":       "#f85149",
    "très_négatif":  "#ff2244",
    "trop_court":    "#484f58",
}
_LABEL_BADGE = {
    "très_positif": "b-pos", "positif": "b-pos",
    "neutre": "b-neu",
    "négatif": "b-neg", "très_négatif": "b-neg",
}

DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", color="#8b949e", size=11),
    margin=dict(l=40, r=20, t=36, b=36),
    xaxis=dict(gridcolor="#1e2730", linecolor="#1e2730", zerolinecolor="#1e2730"),
    yaxis=dict(gridcolor="#1e2730", linecolor="#1e2730", zerolinecolor="#1e2730"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2730"),
)

_PCFG = {"displayModeBar": False, "responsive": True}


def _empty(msg: str, h: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=.5, y=.5,
                       showarrow=False, font=dict(color="#484f58", size=13,
                                                   family="JetBrains Mono"))
    fig.update_layout(**DARK, height=h)
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# CALCUL KPIs
# ════════════════════════════════════════════════════════════════════════════════

def compute_kpis(data: dict) -> dict:
    vs     = _valid_sents(data["sentiments"])
    alerts = data["alerts"]
    dets   = data["detections"]

    scores = [s.get("score", 0) for s in vs]
    labels = [s.get("sentiment", "neutre") for s in vs]
    langs  = [s.get("language", "unknown") for s in vs]

    avg_score  = round(sum(scores) / len(scores), 3) if scores else 0.0
    pos_pct    = int(labels.count("positif") * 100 / len(labels)) if labels else 0
    lang_count = len(set(langs) - {"unknown"})

    recent_alerts = [a for a in alerts if _is_recent(a.get("timestamp",""), 60)]
    sensitive_n   = sum(1 for d in dets if d.get("has_sensitive"))

    return {
        "phrases":       len(vs),
        "avg_score":     avg_score,
        "pos_pct":       pos_pct,
        "lang_count":    lang_count,
        "recent_alerts": len(recent_alerts),
        "total_alerts":  len(alerts),
        "sensitive":     sensitive_n,
        "keystrokes":    len(data["metadata"]),
    }


# ════════════════════════════════════════════════════════════════════════════════
# GRAPHIQUES
# ════════════════════════════════════════════════════════════════════════════════

def chart_timeline(sents: list) -> go.Figure:
    """Évolution temporelle des scores avec couleur par label."""
    vs = _valid_sents(sents)[-100:]
    if not vs:
        return _empty("Tapez des phrases pour voir l'évolution · min 2 mots", 300)

    ts     = [s["timestamp"] for s in vs]
    scores = [s.get("score", 0) for s in vs]
    labels = [s.get("sentiment", "neutre") for s in vs]
    flags  = [s.get("lang_flag", "🌐") for s in vs]
    confs  = [s.get("confidence", 0) for s in vs]
    texts  = [s.get("text", "")[:40] for s in vs]
    colors = [_LABEL_COLOR.get(l, "#8b949e") for l in labels]

    fig = go.Figure()
    # Zones colorées
    fig.add_hrect(y0=.10, y1=1,   fillcolor="#3fb950", opacity=.04, line_width=0)
    fig.add_hrect(y0=-.10, y1=.10, fillcolor="#8b949e", opacity=.03, line_width=0)
    fig.add_hrect(y0=-1, y1=-.10, fillcolor="#f85149", opacity=.04, line_width=0)

    fig.add_trace(go.Scatter(
        x=ts, y=scores, mode="lines+markers",
        line=dict(color="#388bfd", width=1.5, shape="spline", smoothing=.7),
        marker=dict(color=colors, size=9, line=dict(color="#070b14", width=1.5)),
        fill="tozeroy", fillcolor="rgba(56,139,253,0.04)",
        customdata=list(zip(labels, flags, confs, texts)),
        hovertemplate=(
            "<b>%{customdata[1]} %{customdata[0]}</b><br>"
            "Score : %{y:+.4f}<br>"
            "Confiance : %{customdata[2]:.2f}<br>"
            "<i>%{customdata[3]}</i><extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#1e2730", line_width=1)

    # Annotations min/max si assez de données
    if len(scores) >= 5:
        mx, mn = max(scores), min(scores)
        if mx > .15:
            idx = scores.index(mx)
            fig.add_annotation(x=ts[idx], y=mx, text=f"+{mx:.2f}",
                               showarrow=False, yshift=14,
                               font=dict(color="#3fb950", size=10, family="JetBrains Mono"))
        if mn < -.15:
            idx = scores.index(mn)
            fig.add_annotation(x=ts[idx], y=mn, text=f"{mn:.2f}",
                               showarrow=False, yshift=-14,
                               font=dict(color="#f85149", size=10, family="JetBrains Mono"))

    layout = dict(**DARK)
    layout.update(
        title=dict(text=f"📈 Évolution des sentiments — {len(vs)} phrases",
                   font=dict(color="#e6edf3", size=13)),
        yaxis=dict(**DARK["yaxis"], range=[-1.1, 1.1]),
        height=300,
    )
    fig.update_layout(**layout)
    return fig


def chart_lang_bar(sents: list) -> go.Figure:
    """Distribution des langues détectées (barres horizontales)."""
    vs    = _valid_sents(sents)
    langs = [s.get("language", "unknown") for s in vs if s.get("language") != "unknown"]
    if not langs:
        return _empty("Aucune langue détectée", 240)

    counts = Counter(langs).most_common(10)
    flags  = {
        "fr": "🇫🇷", "en": "🇬🇧", "es": "🇪🇸", "de": "🇩🇪",
        "it": "🇮🇹", "pt": "🇵🇹", "nl": "🇳🇱", "ar": "🇸🇦",
        "zh": "🇨🇳", "ja": "🇯🇵", "ko": "🇰🇷", "ru": "🇷🇺",
    }
    labels_plot = [f"{flags.get(l,'🌐')} {l}" for l, _ in counts]
    values      = [c for _, c in counts]

    palette = ["#388bfd","#3fb950","#d29922","#f85149","#7b2fff",
               "#ff6b35","#00d4ff","#ff9f43","#2ed573","#ff6b81"]

    fig = go.Figure(go.Bar(
        x=values, y=labels_plot, orientation="h",
        marker_color=palette[:len(counts)],
        text=values, textposition="outside",
        textfont=dict(color="#8b949e", size=11, family="JetBrains Mono"),
        hovertemplate="%{y}: %{x} phrases<extra></extra>",
    ))
    layout = dict(**DARK)
    layout.update(
        title=dict(text="🌐 Langues détectées", font=dict(color="#e6edf3", size=13)),
        xaxis=dict(**DARK["xaxis"], title="Nb phrases"),
        yaxis=dict(**DARK["yaxis"], autorange="reversed"),
        height=max(200, 60 + len(counts) * 34),
    )
    fig.update_layout(**layout)
    return fig


def chart_label_donut(sents: list) -> go.Figure:
    """Donut répartition des labels."""
    vs     = _valid_sents(sents)
    counts = Counter(s.get("sentiment", "neutre") for s in vs)
    if not counts:
        return _empty("Aucune donnée", 260)

    order  = ["très_positif","positif","neutre","négatif","très_négatif"]
    labels = [l for l in order if l in counts]
    values = [counts[l] for l in labels]
    colors = [_LABEL_COLOR.get(l, "#8b949e") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=.55,
        marker=dict(colors=colors, line=dict(color="#070b14", width=2)),
        textfont=dict(family="JetBrains Mono", size=10, color="#e6edf3"),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    layout = dict(**DARK)
    layout.update(
        title=dict(text="🏷️ Répartition sentiments", font=dict(color="#e6edf3", size=13)),
        showlegend=True, height=270,
        annotations=[dict(text=f"<b>{len(vs)}</b><br>phrases",
                          x=.5, y=.5, showarrow=False,
                          font=dict(color="#e6edf3", size=13, family="JetBrains Mono"))],
    )
    fig.update_layout(**layout)
    return fig


def chart_confidence_hist(sents: list) -> go.Figure:
    """Histogramme des scores de confiance."""
    vs    = _valid_sents(sents)
    confs = [s.get("confidence", 0) for s in vs]
    if not confs:
        return _empty("Aucune donnée", 240)

    fig = go.Figure(go.Histogram(
        x=confs, nbinsx=20, marker_color="#7b2fff", opacity=.75,
        histnorm="probability density",
        hovertemplate="Confiance: %{x:.2f}<br>Densité: %{y:.3f}<extra></extra>",
    ))
    avg = sum(confs) / len(confs)
    fig.add_vline(x=avg, line_dash="dash", line_color="#d29922",
                  annotation_text=f"moy={avg:.2f}",
                  annotation_font=dict(color="#d29922", size=10))
    layout = dict(**DARK)
    layout.update(
        title=dict(text="📊 Distribution de la confiance", font=dict(color="#e6edf3", size=13)),
        xaxis=dict(**DARK["xaxis"], range=[0, 1]),
        height=240,
    )
    fig.update_layout(**layout)
    return fig


def chart_delay_hist(metadata: list) -> go.Figure:
    """Histogramme délais inter-touches."""
    delays = [m["inter_key_delay"] for m in metadata
              if 0.005 < m.get("inter_key_delay", 0) < 2.0]
    if not delays:
        return _empty("Aucune métadonnée de frappe", 240)

    fig = go.Figure(go.Histogram(
        x=delays, nbinsx=40, marker_color="#388bfd", opacity=.7,
        histnorm="probability density",
    ))
    avg = sum(delays) / len(delays)
    fig.add_vline(x=avg, line_dash="dash", line_color="#d29922",
                  annotation_text=f"μ={avg:.3f}s",
                  annotation_font=dict(color="#d29922", size=10))
    layout = dict(**DARK)
    layout.update(
        title=dict(text="⌨️ Délais inter-touches", font=dict(color="#e6edf3", size=13)),
        xaxis=dict(**DARK["xaxis"], title="Délai (s)"),
        height=240,
    )
    fig.update_layout(**layout)
    return fig


def chart_sensitive_donut(dets: list) -> go.Figure:
    """Donut données sensibles."""
    counts: Counter = Counter()
    for d in dets:
        for det in d.get("detections", []):
            counts[det["type"]] += 1
    if not counts:
        return _empty("✅ Aucune donnée sensible détectée", 240)

    palette = ["#f85149","#d29922","#388bfd","#3fb950","#7b2fff",
               "#ff6b35","#00d4ff","#ff9f43"]
    fig = go.Figure(go.Pie(
        labels=list(counts.keys()), values=list(counts.values()), hole=.55,
        marker=dict(colors=palette[:len(counts)], line=dict(color="#070b14", width=2)),
        textfont=dict(family="JetBrains Mono", size=10),
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    layout = dict(**DARK)
    layout.update(
        title=dict(text="🔒 Données sensibles", font=dict(color="#e6edf3", size=13)),
        showlegend=True, height=240,
    )
    fig.update_layout(**layout)
    return fig


def chart_heatmap(metadata: list) -> go.Figure:
    """Heatmap activité horaire."""
    if not metadata:
        return _empty("Aucune métadonnée", 240)
    days = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
    mat  = [[0]*24 for _ in range(7)]
    for m in metadata:
        try:
            dt = datetime.fromtimestamp(m["timestamp"])
            mat[dt.weekday()][dt.hour] += 1
        except Exception:
            pass
    fig = go.Figure(go.Heatmap(
        z=mat, x=list(range(24)), y=days,
        colorscale=[[0,"#010409"],[.3,"#0d2d4e"],[.7,"#1a4d8a"],[1,"#388bfd"]],
        hoverongaps=False,
        hovertemplate="Jour:%{y}  H:%{x}h  Frappes:%{z}<extra></extra>",
        showscale=True,
        colorbar=dict(bgcolor="rgba(0,0,0,0)", tickfont=dict(color="#8b949e")),
    ))
    layout = dict(**DARK)
    layout.update(
        title=dict(text="🕐 Activité horaire", font=dict(color="#e6edf3", size=13)),
        height=240,
    )
    fig.update_layout(**layout)
    return fig


def chart_anomaly(alerts: list) -> go.Figure:
    """Scatter anomalies."""
    if not alerts:
        return _empty("✅ Aucune anomalie détectée", 240)
    ts     = [a["timestamp"] for a in alerts]
    scores = [a.get("score", -.5) for a in alerts]
    recent = [_is_recent(a.get("timestamp",""), 60) for a in alerts]
    colors = ["#f85149" if r else "#8b949e" for r in recent]
    fig = go.Figure(go.Scatter(
        x=ts, y=scores, mode="markers",
        marker=dict(color=colors, size=11, symbol="x-thin",
                    line=dict(color=colors, width=2.5)),
        hovertemplate="<b>%{x}</b><br>Score:%{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#1e2730")
    layout = dict(**DARK)
    layout.update(
        title=dict(text="⚠️ Anomalies comportementales", font=dict(color="#e6edf3", size=13)),
        height=240,
    )
    fig.update_layout(**layout)
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# COMPOSANTS VISUELS
# ════════════════════════════════════════════════════════════════════════════════

def render_statusbar(data: dict, refresh: int) -> None:
    age     = data["age_log"]
    live    = age < 60
    cls     = "live" if live else "stale"
    txt     = "EN DIRECT" if live else "KEYLOGGER INACTIF"
    age_str = f"{int(age)}s" if age < 3600 else "∞"

    st.markdown(f"""
    <div class="statusbar">
        <span class="{cls}">{txt}</span>
        <span style="color:#484f58">
            log: <span style="color:{'#3fb950' if age<30 else '#d29922' if age<120 else '#f85149'}">{age_str}</span> · 
            sentiments: <span style="color:{'#3fb950' if data['age_sent']<30 else '#d29922' if data['age_sent']<120 else '#f85149'}">{int(data['age_sent'])}s</span>
        </span>
        <span style="color:#484f58">rafraîch. {refresh}s · {data['ts'].strftime('%H:%M:%S')}</span>
    </div>""", unsafe_allow_html=True)

    if not live:
        st.warning(
            "⚠️ **Keylogger inactif.** Lancez `python keylogger.py` pour alimenter le dashboard en temps réel.",
            icon=None
        )


def render_kpis(kpis: dict) -> None:
    sc = kpis["avg_score"]
    sc_col = "#3fb950" if sc > .1 else "#f85149" if sc < -.1 else "#8b949e"
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-icon">⌨️</div>
            <div class="kpi-value">{kpis['keystrokes']:,}</div>
            <div class="kpi-label">Frappes capturées</div>
            <div class="kpi-sub">{kpis['phrases']} phrases analysées</div>
        </div>
        <div class="kpi-card {'green' if sc>=0 else 'red'}">
            <div class="kpi-icon">🧠</div>
            <div class="kpi-value" style="color:{sc_col}">{sc:+.3f}</div>
            <div class="kpi-label">Score sentiment moyen</div>
            <div class="kpi-sub">{kpis['pos_pct']}% positif</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-icon">🌐</div>
            <div class="kpi-value">{kpis['lang_count']}</div>
            <div class="kpi-label">Langues détectées</div>
            <div class="kpi-sub">Analyse multilingue</div>
        </div>
        <div class="kpi-card {'red' if kpis['recent_alerts']>0 else 'green'}">
            <div class="kpi-icon">⚠️</div>
            <div class="kpi-value">{kpis['recent_alerts']}</div>
            <div class="kpi-label">Anomalies (1h)</div>
            <div class="kpi-sub">Total: {kpis['total_alerts']}</div>
        </div>
        <div class="kpi-card {'red' if kpis['sensitive']>0 else 'green'}">
            <div class="kpi-icon">🔒</div>
            <div class="kpi-value">{kpis['sensitive']}</div>
            <div class="kpi-label">Données sensibles</div>
            <div class="kpi-sub">Détections session</div>
        </div>
    </div>""", unsafe_allow_html=True)


def render_header(kpis: dict, ts: datetime) -> None:
    sc   = kpis["avg_score"]
    mood = "😊" if sc > .3 else "😔" if sc < -.3 else "😐"
    st.markdown(f"""
    <div class="dash-header">
        <h1>🌐 AI Keylogger · Sentiment ML — FR/EN/ES/DE/IT</h1>
        <p>TP1 IA & Cybersécurité · {ts.strftime('%d/%m/%Y %H:%M:%S')} ·
           Humeur globale : {mood} {sc:+.3f} · {kpis['phrases']} phrases · {kpis['lang_count']} langue(s)</p>
    </div>""", unsafe_allow_html=True)


def render_sent_table(sents: list, n: int = 15) -> None:
    st.markdown('<div class="sec">🧠 Dernières analyses — multilingue temps réel</div>',
                unsafe_allow_html=True)
    vs = _valid_sents(sents)[-n:]
    if not vs:
        st.markdown('<div style="color:#484f58;font-family:JetBrains Mono,monospace;font-size:.82em;">Tapez des phrases (min. 2 mots) pour voir les analyses ici.</div>',
                    unsafe_allow_html=True)
        return

    html = ""
    for s in reversed(vs):
        label  = s.get("sentiment", "neutre")
        score  = s.get("score", 0)
        lang   = s.get("language", "?")
        flag   = s.get("lang_flag", "🌐")
        conf   = s.get("confidence", 0)
        text   = s.get("text", "")[:60] + ("…" if len(s.get("text","")) > 60 else "")
        ts_str = s.get("timestamp", "")[:16]
        color  = _LABEL_COLOR.get(label, "#8b949e")
        badge  = _LABEL_BADGE.get(label, "b-neu")
        bar    = abs(score) * 100
        bar_c  = "#3fb950" if score > 0 else "#f85149"
        conf_w = int(conf * 100)
        conf_c = "#3fb950" if conf > .6 else "#d29922" if conf > .3 else "#f85149"

        html += f"""
        <div class="sent-row">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <span class="sent-text">{flag} {text}</span>
                <span class="badge {badge}" style="white-space:nowrap;margin-left:8px;">{score:+.3f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px;">
                <div style="flex:1;margin-right:12px;">
                    <div class="pbar-wrap">
                        <div class="pbar-fill" style="width:{bar:.0f}%;background:{bar_c};"></div>
                    </div>
                </div>
                <span class="badge {badge}" style="font-size:.65em;">{label}</span>
                <span class="lang-badge">{lang}</span>
                <span style="font-size:.65em;color:{conf_c};margin-left:6px;font-family:'JetBrains Mono',monospace;">conf:{conf_w}%</span>
                <span style="font-size:.65em;color:#484f58;margin-left:8px;">{ts_str}</span>
            </div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_detections(dets: list) -> None:
    st.markdown('<div class="sec">🔒 Données sensibles récentes</div>', unsafe_allow_html=True)
    recent = [d for d in dets if d.get("has_sensitive")][-8:]
    if not recent:
        st.markdown('<div class="det-row"><span style="color:#3fb950;font-family:JetBrains Mono,monospace;font-size:.82em;">✅ Aucune donnée sensible détectée</span></div>',
                    unsafe_allow_html=True)
        return
    for r in reversed(recent):
        ts_str = r.get("timestamp", "")[:19]
        for det in r.get("detections", []):
            dtype  = det["type"].replace("_"," ").upper()
            method = det.get("method","regex")
            mb     = "b-warn" if method == "regex" else "b-info"
            st.markdown(f"""
            <div class="det-row">
                <div>
                    <span class="badge {mb}">{dtype}</span>
                    <span style="font-size:.72em;color:#484f58;margin-left:8px;">[{method}] len={det.get('length',0)}</span>
                </div>
                <div class="det-time">{ts_str}</div>
            </div>""", unsafe_allow_html=True)


def render_alerts(alerts: list) -> None:
    st.markdown('<div class="sec">🚨 Anomalies comportementales</div>', unsafe_allow_html=True)
    recent = [a for a in alerts if _is_recent(a.get("timestamp",""), 120)][-8:]
    if not recent:
        st.markdown('<div class="det-row"><span style="color:#3fb950;font-family:JetBrains Mono,monospace;font-size:.82em;">✅ Aucune anomalie récente</span></div>',
                    unsafe_allow_html=True)
        return
    for a in reversed(recent):
        sc  = a.get("score", 0)
        sev = "CRITIQUE" if sc < -.6 else "ALERTE"
        bc  = "b-crit" if sev == "CRITIQUE" else "b-warn"
        ts_str = a.get("timestamp","")[:19]
        st.markdown(f"""
        <div class="det-row">
            <div>
                <span class="badge {bc}">{sev}</span>
                <span style="font-size:.78em;color:#e6edf3;margin-left:10px;font-family:'JetBrains Mono',monospace;">score={sc:.4f}</span>
            </div>
            <div class="det-time">{ts_str}</div>
        </div>""", unsafe_allow_html=True)


def render_log(log_lines: list, n: int) -> None:
    st.markdown('<div class="sec">📋 Log temps réel</div>', unsafe_allow_html=True)
    if not log_lines:
        st.markdown('<div class="log-box" style="color:#484f58;">Aucun log — lancez keylogger.py</div>',
                    unsafe_allow_html=True)
        return
    rows = ""
    for line in log_lines[-n:]:
        safe = line.rstrip().replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        if safe.startswith("[20"):
            rows += f'<div style="color:#388bfd;margin-top:4px;">{safe}</div>'
        elif "—" in safe and len(safe.strip("—")) == 0:
            rows += '<hr style="border-color:#1e2730;margin:4px 0;">'
        else:
            rows += f'<div style="color:#e6edf3;">{safe}</div>'
    st.markdown(f'<div class="log-box">{rows}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

def render_sidebar(kpis: dict) -> dict:
    with st.sidebar:
        st.markdown("""
        <div style="padding:12px 0;border-bottom:1px solid #1e2730;margin-bottom:16px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:.72em;color:#8b949e;
                        text-transform:uppercase;letter-spacing:.1em;">AI Keylogger</div>
            <div style="font-size:1.1em;font-weight:700;color:#e6edf3;">🌐 Multilingue v4 · ML</div>
        </div>""", unsafe_allow_html=True)

        view    = st.selectbox("Vue", ["Vue globale","Sentiments","Anomalies",
                                        "Données sensibles","Logs bruts"])
        refresh = st.slider("Rafraîchissement (s)", 3, 60, 5)
        n_sent  = st.slider("Phrases affichées", 10, 50, 20)
        n_log   = st.slider("Lignes de log", 20, 200, 60)

        st.markdown("---")
        if st.button("🔄 Forcer actualisation", use_container_width=True):
            st.rerun()
        if st.button("🗑️ Vider sentiments.json", use_container_width=True):
            p = DATA / "sentiments.json"
            if p.exists():
                p.write_text("[]", encoding="utf-8")
            st.success("sentiments.json réinitialisé")

        st.markdown("---")
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:.75em;color:#8b949e;">
            <div style="margin-bottom:4px;">📊 {kpis['phrases']} phrases analysées</div>
            <div style="margin-bottom:4px;">🌐 {kpis['lang_count']} langue(s) détectée(s)</div>
            <div style="margin-bottom:4px;">⌨️ {kpis['keystrokes']:,} frappes</div>
            <div>🔒 {kpis['sensitive']} données sensibles</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-family:'JetBrains Mono',monospace;font-size:.7em;color:#484f58;">
            <b style="color:#8b949e;">Langues supportées :</b><br>
            🇫🇷 FR · 🇬🇧 EN · 🇪🇸 ES · 🇩🇪 DE · 🇮🇹 IT<br>
            <br><i>Modèle ML — char N-gram<br>sans lexique ni dictionnaire</i>
        </div>""", unsafe_allow_html=True)

    return {"view": view, "refresh": refresh, "n_sent": n_sent, "n_log": n_log}


# ════════════════════════════════════════════════════════════════════════════════
# VUES
# ════════════════════════════════════════════════════════════════════════════════

def view_global(data: dict, cfg: dict) -> None:
    render_statusbar(data, cfg["refresh"])

    # Ligne 1 : timeline (large) + langue (petit)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(chart_timeline(data["sentiments"]),
                        use_container_width=True, config=_PCFG)
    with c2:
        st.plotly_chart(chart_lang_bar(data["sentiments"]),
                        use_container_width=True, config=_PCFG)

    # Ligne 2 : donut sentiments + confiance + activité
    c3, c4, c5 = st.columns(3)
    with c3:
        st.plotly_chart(chart_label_donut(data["sentiments"]),
                        use_container_width=True, config=_PCFG)
    with c4:
        st.plotly_chart(chart_confidence_hist(data["sentiments"]),
                        use_container_width=True, config=_PCFG)
    with c5:
        st.plotly_chart(chart_heatmap(data["metadata"]),
                        use_container_width=True, config=_PCFG)

    # Ligne 3 : table sentiments + alertes
    c6, c7 = st.columns([3, 2])
    with c6:
        render_sent_table(data["sentiments"], cfg["n_sent"])
    with c7:
        render_alerts(data["alerts"])
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        render_detections(data["detections"])

    # Ligne 4 : log + délais
    c8, c9 = st.columns([3, 2])
    with c8:
        render_log(data["log_lines"], cfg["n_log"])
    with c9:
        st.plotly_chart(chart_delay_hist(data["metadata"]),
                        use_container_width=True, config=_PCFG)


def view_sentiments(data: dict, cfg: dict) -> None:
    render_statusbar(data, cfg["refresh"])
    st.plotly_chart(chart_timeline(data["sentiments"]),
                    use_container_width=True, config=_PCFG)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(chart_label_donut(data["sentiments"]),
                        use_container_width=True, config=_PCFG)
    with c2:
        st.plotly_chart(chart_lang_bar(data["sentiments"]),
                        use_container_width=True, config=_PCFG)
    with c3:
        st.plotly_chart(chart_confidence_hist(data["sentiments"]),
                        use_container_width=True, config=_PCFG)

    # Stats rapides par langue
    vs    = _valid_sents(data["sentiments"])
    langs = list({s.get("language","?") for s in vs if s.get("language","?") != "unknown"})
    if langs:
        st.markdown('<div class="sec" style="margin-top:16px;">📊 Statistiques par langue</div>',
                    unsafe_allow_html=True)
        cols = st.columns(min(len(langs), 5))
        flags = {"fr":"🇫🇷","en":"🇬🇧","es":"🇪🇸","de":"🇩🇪","it":"🇮🇹",
                 "pt":"🇵🇹","nl":"🇳🇱","ar":"🇸🇦","zh":"🇨🇳","ja":"🇯🇵"}
        for col, lang in zip(cols, langs[:5]):
            subset = [s for s in vs if s.get("language") == lang]
            scores = [s.get("score", 0) for s in subset]
            avg    = round(sum(scores) / len(scores), 3) if scores else 0
            flag   = flags.get(lang, "🌐")
            c      = "#3fb950" if avg > 0 else "#f85149"
            col.markdown(f"""
            <div style="background:#0d1117;border:1px solid #1e2730;border-radius:10px;
                        padding:16px;text-align:center;">
                <div style="font-size:2em;">{flag}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.4em;
                            font-weight:800;color:{c};">{avg:+.3f}</div>
                <div style="font-size:.72em;color:#8b949e;margin-top:4px;">{lang.upper()} · {len(subset)} phrases</div>
            </div>""", unsafe_allow_html=True)

    render_sent_table(data["sentiments"], cfg["n_sent"])


def view_anomalies(data: dict, cfg: dict) -> None:
    render_statusbar(data, cfg["refresh"])
    st.plotly_chart(chart_anomaly(data["alerts"]),
                    use_container_width=True, config=_PCFG)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_delay_hist(data["metadata"]),
                        use_container_width=True, config=_PCFG)
    with c2:
        render_alerts(data["alerts"])


def view_sensitive(data: dict, cfg: dict) -> None:
    render_statusbar(data, cfg["refresh"])
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_sensitive_donut(data["detections"]),
                        use_container_width=True, config=_PCFG)
    with c2:
        render_detections(data["detections"])


def view_logs(data: dict, cfg: dict) -> None:
    render_statusbar(data, cfg["refresh"])
    render_log(data["log_lines"], cfg["n_log"])


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Chargement sans cache — données fraîches garanties
    data = load_all()
    kpis = compute_kpis(data)
    cfg  = render_sidebar(kpis)

    render_header(kpis, data["ts"])
    render_kpis(kpis)

    view = cfg["view"]
    if   view == "Vue globale":        view_global(data, cfg)
    elif view == "Sentiments":         view_sentiments(data, cfg)
    elif view == "Anomalies":          view_anomalies(data, cfg)
    elif view == "Données sensibles":  view_sensitive(data, cfg)
    elif view == "Logs bruts":         view_logs(data, cfg)

    # ── Compteur de rafraîchissement visible ──────────────────────────────
    placeholder = st.empty()
    for remaining in range(cfg["refresh"], 0, -1):
        placeholder.markdown(
            f"<div style='text-align:center;color:#1e2730;font-family:JetBrains Mono,monospace;"
            f"font-size:.7em;margin-top:12px;padding-bottom:8px;'>"
            f"⏱ Rafraîchissement dans {remaining}s</div>",
            unsafe_allow_html=True,
        )
        time.sleep(1)
    placeholder.empty()
    st.rerun()


if __name__ == "__main__":
    main()
