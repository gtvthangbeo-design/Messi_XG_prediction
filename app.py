from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Messi xG Lab", page_icon="⚽", layout="wide")

# =====================================================================
# 1. MODEL
# =====================================================================
MODEL_PATH = Path(__file__).parent / "messi_xg_rf_full.pkl"


@st.cache_resource
def load_model():
    data = joblib.load(MODEL_PATH)
    return data["model"], data["features"]


model, FEATURES = load_model()

YARD_TO_M = 0.9144  # toạ độ sân kiểu StatsBomb (120x80) tính bằng yard


# =====================================================================
# 2. HÌNH HỌC (vector hoá để dùng cho cả lưới click trên sân)
# =====================================================================
def goal_distance(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.sqrt((120 - x) ** 2 + (40 - y) ** 2)


def goal_angle(x, y):
    """Góc mở khung thành (độ) giữa hai cột dọc (y=36 và y=44)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dx = 120 - x
    cross = dx * 8.0
    dot = dx**2 + (36 - y) * (44 - y)
    return np.degrees(np.arctan2(np.abs(cross), dot))


@st.cache_data
def click_grid():
    xs, ys = np.meshgrid(np.arange(60, 121, 1.0), np.arange(0, 81, 1.0))
    xs, ys = xs.ravel(), ys.ravel()
    custom = np.column_stack([goal_distance(xs, ys) * YARD_TO_M, goal_angle(xs, ys)])
    return xs, ys, custom


# =====================================================================
# 3. NGÔN NGỮ
# =====================================================================
TXT = {
    "vi": {
        "dark": "🌙 Chế độ tối",
        "hero_title": "Messi xG Lab",
        "eyebrow": "AI · 17 NĂM DỮ LIỆU CỦA MESSI",
        "hero_sub": "Chạm vào sân để đặt vị trí cú sút, chọn tình huống và xem xác suất thành bàn thay đổi ngay lập tức.",
        "corner_mode_title": "Kiểu cú sút từ phạt góc",
        "corner_mode": {"after": "Sút trong vòng cấm", "direct": "Sút thẳng từ góc"},
        "side_title": "Chọn góc đá",
        "side": {"L": "↖ Góc trái", "R": "Góc phải ↗"},
        "direct_note": "Bóng đặt tại cờ góc, cú sút đi thẳng vào khung thành",
        "pitch_title": "Sân bóng · Nửa sân tấn công",
        "pitch_hint": "👆 Chạm vào bất kỳ điểm nào trên sân để đặt vị trí cú sút",
        "goal": "KHUNG THÀNH",
        "pos": "Vị trí",
        "pen_note": "Phạt đền: bóng được đặt cố định tại chấm 11m",
        "sit_title": "Tình huống",
        "sit": {"open": "⚽ Bóng sống", "fk": "🎯 Đá phạt", "corner": "🚩 Phạt góc", "pen": "🥅 Phạt đền"},
        "hint": {
            "open": "Bóng đang trong cuộc: tính cả hậu vệ chắn phễu sút và hậu vệ áp sát.",
            "fk": "Đá phạt trực tiếp: chỉ tính số hậu vệ trong hàng rào, không có áp sát hay sút một chạm.",
            "corner": "Phạt góc: chạm vào sân để chọn nơi dứt điểm trong vòng cấm, hoặc chuyển sang sút thẳng từ góc. Không tính hậu vệ chắn hay áp sát.",
            "pen": "Phạt đền: chỉ cần chọn chân sút, mọi yếu tố khác đã cố định.",
        },
        "sec_ctx": "Bối cảnh",
        "sec_def": "Áp lực phòng ngự",
        "sec_tech": "Kỹ thuật",
        "pattern": "Bóng đến từ tình huống nào?",
        "pat": {
            "Regular Play": "Phối hợp thông thường",
            "From Counter": "Phản công nhanh",
            "From Throw In": "Sau tình huống ném biên",
            "From Keeper": "Từ thủ môn phát bóng",
            "From Goal Kick": "Từ quả phát bóng lên",
            "Other": "Khác",
        },
        "cone": "Hậu vệ chắn phễu sút",
        "cone_help": "Số cầu thủ phòng ngự nằm giữa bóng và khung thành.",
        "wall": "Số hậu vệ trong hàng rào",
        "close": "Hậu vệ áp sát (<2m)",
        "pressure": "Bị tì đè / áp lực mạnh",
        "first": "Sút một chạm",
        "body": "Bộ phận dứt điểm",
        "bod": {"Left Foot": "Chân trái", "Right Foot": "Chân phải", "Head": "Đánh đầu", "Other": "Bộ phận khác (ngực, vai...)"},
        "tech": "Kỹ thuật dứt điểm",
        "tec": {
            "Normal": "Thông thường",
            "Half Volley": "Bán lăng (half volley)",
            "Volley": "Vô lê",
            "Lob": "Đá bóng bổng (lốp)",
            "Diving Header": "Đánh đầu lao người",
            "Backheel": "Đánh gót",
            "Overhead Kick": "Xe đạp chổng ngược",
        },
        "xg": "XÁC SUẤT THÀNH BÀN (xG)",
        "dist": "Khoảng cách tới gôn",
        "angle": "Góc mở khung thành",
        "per100": "bàn / 100 cú sút",
        "verdicts": ["Cơ hội thấp", "Cơ hội trung bình", "Cơ hội tốt", "Rất nguy hiểm", "Gần như chắc chắn"],
        "hover_d": "Khoảng cách",
        "hover_a": "Góc sút",
        "foot": "Toạ độ theo hệ 120×80 (StatsBomb); khoảng cách đã quy đổi sang mét. "
        "Kết quả là ước tính của mô hình, không phải dự đoán chắc chắn.",
    },
    "en": {
        "dark": "🌙 Dark mode",
        "hero_title": "Messi xG Lab",
        "eyebrow": "AI · 17 YEARS OF MESSI DATA",
        "hero_sub": "Tap the pitch to place the shot, pick the situation and watch the scoring probability update instantly.",
        "corner_mode_title": "Corner shot type",
        "corner_mode": {"after": "Shot in the box", "direct": "Direct from corner"},
        "side_title": "Corner taken from",
        "side": {"L": "↖ Left corner", "R": "Right corner ↗"},
        "direct_note": "Ball placed on the corner flag, shot goes straight at goal",
        "pitch_title": "Pitch · Attacking half",
        "pitch_hint": "👆 Tap anywhere on the pitch to set the shot location",
        "goal": "GOAL",
        "pos": "Position",
        "pen_note": "Penalty: the ball is fixed on the penalty spot",
        "sit_title": "Situation",
        "sit": {"open": "⚽ Open play", "fk": "🎯 Free kick", "corner": "🚩 Corner", "pen": "🥅 Penalty"},
        "hint": {
            "open": "Live play: both blockers in the shooting cone and close defenders matter.",
            "fk": "Direct free kick: only the number of defenders in the wall counts. No close marking or first-time shot.",
            "corner": "Corner: tap the pitch to set where the shot is taken in the box, or switch to a direct shot from the corner. Blockers and close defenders are not used.",
            "pen": "Penalty: just choose the shooting foot, everything else is fixed.",
        },
        "sec_ctx": "Context",
        "sec_def": "Defensive pressure",
        "sec_tech": "Technique",
        "pattern": "How did the move start?",
        "pat": {
            "Regular Play": "Regular play",
            "From Counter": "Counter-attack",
            "From Throw In": "From a throw-in",
            "From Keeper": "From the keeper",
            "From Goal Kick": "From a goal kick",
            "Other": "Other",
        },
        "cone": "Blocking defenders",
        "cone_help": "Defenders standing between the ball and the goal.",
        "wall": "Defenders in the wall",
        "close": "Close defenders (<2m)",
        "pressure": "Under heavy pressure",
        "first": "First-time shot",
        "body": "Body part",
        "bod": {"Left Foot": "Left foot", "Right Foot": "Right foot", "Head": "Header", "Other": "Other (chest, shoulder...)"},
        "tech": "Shot technique",
        "tec": {
            "Normal": "Normal",
            "Half Volley": "Half volley",
            "Volley": "Volley",
            "Lob": "Lob",
            "Diving Header": "Diving header",
            "Backheel": "Backheel",
            "Overhead Kick": "Overhead kick",
        },
        "xg": "SCORING PROBABILITY (xG)",
        "dist": "Distance to goal",
        "angle": "Goal opening angle",
        "per100": "goals / 100 shots",
        "verdicts": ["Low chance", "Moderate chance", "Good chance", "Very dangerous", "Almost certain"],
        "hover_d": "Distance",
        "hover_a": "Angle",
        "foot": "Coordinates use the 120×80 (StatsBomb) system; distances are converted to metres. "
        "Results are model estimates, not guarantees.",
    },
}

# =====================================================================
# 4. GIAO DIỆN: CSS SÁNG / TỐI
# =====================================================================
PALETTE = {
    True: {
        "bg": "#080e1c", "glow1": "rgba(34,197,94,.16)", "glow2": "rgba(56,189,248,.13)",
        "card": "rgba(16,25,45,.86)", "card-solid": "#121c33", "border": "rgba(255,255,255,.08)",
        "text": "#e8eefc", "muted": "#9aa9c6", "input": "#18243f", "accent": "#22c55e",
        "accent2": "#38bdf8", "shadow": "0 12px 40px rgba(0,0,0,.40)", "hintbg": "rgba(34,197,94,.09)",
    },
    False: {
        "bg": "#edf3ef", "glow1": "rgba(34,197,94,.20)", "glow2": "rgba(56,189,248,.18)",
        "card": "rgba(255,255,255,.95)", "card-solid": "#ffffff", "border": "rgba(15,23,42,.08)",
        "text": "#0f172a", "muted": "#5a6a85", "input": "#eef2f7", "accent": "#16a34a",
        "accent2": "#0284c7", "shadow": "0 12px 32px rgba(15,23,42,.10)", "hintbg": "rgba(22,163,74,.09)",
    },
}

BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
.stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp div[data-testid="stMarkdownContainer"] {
  font-family: 'Be Vietnam Pro', 'Segoe UI', sans-serif;
}
.stApp {
  background:
    radial-gradient(1100px 560px at 6% -8%, var(--glow1), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, var(--glow2), transparent 55%),
    var(--bg);
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1240px; }

.stApp [data-testid="stMarkdownContainer"] p, .stApp label p { color: var(--text); }
.stApp [data-testid="stWidgetLabel"] p { color: var(--muted); font-weight: 600; font-size: .84rem; }
.stApp [data-testid="stCheckbox"] label p, .stApp [data-testid="stToggle"] label p { color: var(--text); font-weight: 500; font-size: .92rem; }
.stApp [data-testid="stTickBarMin"], .stApp [data-testid="stTickBarMax"] { color: var(--muted); }

/* ---------- thẻ ---------- */
.st-key-card_pitch, .st-key-card_inputs {
  background: var(--card); border: 1px solid var(--border); border-radius: 24px;
  padding: 1.25rem 1.4rem 1.4rem 1.4rem; box-shadow: var(--shadow); backdrop-filter: blur(12px);
}
.brand { font-weight: 800; font-size: 1.1rem; color: var(--text); letter-spacing: .01em; }
.card-title { font-weight: 800; font-size: 1.12rem; color: var(--text); margin-bottom: .55rem; }
.card-sub { color: var(--muted); font-size: .86rem; margin: .1rem 0 .7rem 0; }
.sec { display: flex; align-items: center; gap: .55rem; font-weight: 800; font-size: .74rem; letter-spacing: .1em;
       text-transform: uppercase; color: var(--accent); margin: 1.15rem 0 .35rem 0; }
.sec::after { content: ""; flex: 1; height: 1px; background: var(--border); }
.hint { background: var(--hintbg); border-left: 3px solid var(--accent); padding: .6rem .9rem;
        border-radius: 12px; color: var(--muted); font-size: .86rem; margin: .6rem 0 .1rem 0; line-height: 1.45; }
.chip { display: inline-block; background: var(--input); border: 1px solid var(--border);
        border-radius: 999px; padding: .25rem .8rem; font-size: .8rem; margin: 0 .4rem .6rem 0; color: var(--text); font-weight: 500; }
.foot { color: var(--muted); font-size: .78rem; text-align: center; margin-top: 1.4rem; }

/* ---------- banner kết quả ---------- */
.hero {
  display: grid; grid-template-columns: 1.05fr 1fr; gap: 2rem; align-items: center;
  border-radius: 28px; padding: 1.9rem 2.1rem; margin: .4rem 0 1.2rem 0; position: relative; overflow: hidden;
  background:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='520' height='340' viewBox='0 0 520 340' fill='none' stroke='white' stroke-opacity='.06' stroke-width='2'%3E%3Crect x='20' y='20' width='480' height='300'/%3E%3Crect x='20' y='90' width='110' height='160'/%3E%3Crect x='390' y='90' width='110' height='160'/%3E%3Ccircle cx='260' cy='170' r='60'/%3E%3Cpath d='M260 20V320'/%3E%3C/svg%3E") right -40px center / 520px no-repeat,
    linear-gradient(120deg, #04302b 0%, #065f46 42%, #0b7a8f 100%);
  box-shadow: 0 18px 50px rgba(6,95,70,.35);
}
.eyebrow { display: inline-block; font-size: .72rem; font-weight: 800; letter-spacing: .12em; color: #a7f3d0;
           background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.16); padding: .3rem .8rem; border-radius: 999px; }
.hero-title { color: #fff !important; font-size: 2.7rem; font-weight: 800; line-height: 1.08; margin: .9rem 0 .6rem 0; letter-spacing: -.01em; }
.hero-sub { color: rgba(255,255,255,.86) !important; font-size: 1.02rem; line-height: 1.55; max-width: 470px; }
.glass { background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.18); border-radius: 22px;
         padding: 1.2rem 1.35rem; backdrop-filter: blur(10px); }
.res-label { color: rgba(255,255,255,.75); font-weight: 700; font-size: .72rem; letter-spacing: .12em; }
.res-row { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; margin: .1rem 0 .2rem 0; }
.res-value { font-size: 4.3rem; font-weight: 800; line-height: 1.05; color: #fff; text-shadow: 0 4px 24px rgba(0,0,0,.25); }
.res-value small { font-size: 1.8rem; font-weight: 700; opacity: .9; }
.pill { display: inline-block; padding: .32rem .9rem; border-radius: 999px; font-weight: 700; font-size: .84rem; }
.pill.l0 { background: rgba(248,113,113,.28); color: #fecaca; }
.pill.l1 { background: rgba(251,191,36,.26); color: #fde68a; }
.pill.l2 { background: rgba(163,230,53,.26); color: #ecfccb; }
.pill.l3 { background: rgba(74,222,128,.28); color: #dcfce7; }
.pill.l4 { background: rgba(56,189,248,.30); color: #e0f2fe; }
.bar { position: relative; height: 11px; border-radius: 999px; overflow: hidden; margin: .8rem 0 1rem 0;
       background: linear-gradient(90deg, #ef4444 0%, #f59e0b 30%, #22c55e 65%, #38bdf8 100%); }
.bar .mask { position: absolute; top: 0; bottom: 0; right: 0; background: #0a3140; }
.tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: .65rem; }
.tile { background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.13); border-radius: 14px; padding: .6rem .8rem; }
.tile .v { font-size: 1.22rem; font-weight: 800; color: #fff; }
.tile .l { font-size: .72rem; color: rgba(255,255,255,.72); line-height: 1.25; }

/* ---------- ô nhập ---------- */
div[data-baseweb="select"] > div, div[data-baseweb="select"] > div:hover, div[data-baseweb="select"] > div:focus-within { background: var(--input) !important; background-color: var(--input) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
div[data-baseweb="select"] * { color: var(--text) !important; }
div[data-baseweb="select"] svg { fill: var(--muted) !important; }
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] [data-baseweb="menu"] { background: var(--card-solid) !important; }
div[data-baseweb="popover"] li { color: var(--text) !important; background: transparent !important; }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] { background: var(--input) !important; }

/* ô chọn kiểu mới (react-aria) + danh sách thả xuống */
[data-testid="stSelectbox"] [role="group"] { background: var(--input) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
[data-testid="stSelectbox"] input { color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; background: transparent !important; }
[data-testid="stSelectbox"] button, [data-testid="stSelectbox"] svg { color: var(--muted) !important; fill: var(--muted) !important; background: transparent !important; }
[data-testid="stSelectboxVirtualDropdown"] { background: var(--card-solid) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"] { color: var(--text) !important; background: transparent !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover, [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
[data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"] { background: var(--input) !important; }

/* ---------- radio dạng viên thuốc (key bắt đầu bằng pill_) ---------- */
[class*="st-key-pill_"] [data-testid="stRadioGroup"] { gap: .45rem; flex-wrap: wrap; }
[class*="st-key-pill_"] label[data-testid="stRadioOption"] {
  background: var(--input); border: 1px solid var(--border); border-radius: 999px;
  padding: .42rem 1.05rem; margin: 0; cursor: pointer; transition: all .15s ease; }
[class*="st-key-pill_"] label[data-testid="stRadioOption"]:hover { border-color: var(--accent); transform: translateY(-1px); }
[class*="st-key-pill_"] label[data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) { display: none; }
[class*="st-key-pill_"] label[data-testid="stRadioOption"][data-selected="true"] {
  background: linear-gradient(135deg, var(--accent), var(--accent2)); border-color: transparent;
  box-shadow: 0 6px 16px rgba(34,197,94,.28); }
[class*="st-key-pill_"] label[data-testid="stRadioOption"][data-selected="true"] p { color: #fff !important; font-weight: 700; }
[class*="st-key-pill_"] label[data-testid="stRadioOption"] p { font-size: .9rem; white-space: nowrap; }
.st-key-pill_lang { display: flex; justify-content: flex-end; }

@media (max-width: 900px) {
  .hero { grid-template-columns: 1fr; padding: 1.4rem; gap: 1.2rem; }
  .hero-title { font-size: 2rem; } .res-value { font-size: 3.4rem; } .tiles { grid-template-columns: 1fr 1fr 1fr; }
}
"""


def inject_css(dark: bool):
    root = ":root{" + ";".join(f"--{k}:{v}" for k, v in PALETTE[dark].items()) + "}"
    st.markdown(f"<style>{root}{BASE_CSS}</style>", unsafe_allow_html=True)


# =====================================================================
# 5. VẼ SÂN BÓNG (có thể bấm chọn vị trí)
# =====================================================================
def pitch_line(fig, xs, ys, color, width=2):
    # Trục ngang của biểu đồ = toạ độ Y, trục dọc = toạ độ X (khung thành ở trên)
    fig.add_trace(go.Scatter(x=ys, y=xs, mode="lines", line=dict(color=color, width=width),
                             hoverinfo="skip", showlegend=False))


def make_pitch(dark, pos, interactive, T):
    x0, y0 = pos
    stripe_a, stripe_b = ("#14502b", "#1b6a39") if dark else ("#5dbb6c", "#4eab5e")
    line_c = "rgba(255,255,255,.92)"
    fig = go.Figure()

    # nền sân bo tròn + sọc cỏ
    fig.add_shape(type="rect", x0=-2.5, x1=82.5, y0=58.8, y1=121.2, layer="below",
                  fillcolor="rgba(0,0,0,.18)" if dark else "rgba(15,23,42,.06)", line_width=0)
    for i in range(5):
        fig.add_shape(type="rect", x0=0, x1=80, y0=60 + 12 * i, y1=72 + 12 * i, layer="below",
                      fillcolor=stripe_a if i % 2 == 0 else stripe_b, line_width=0)
    # lưới khung thành
    goal_c = "rgba(255,255,255,.92)" if dark else "rgba(15,23,42,.80)"
    fig.add_shape(type="rect", x0=36, x1=44, y0=120, y1=123.4, layer="below",
                  fillcolor="rgba(255,255,255,.22)" if dark else "rgba(15,23,42,.10)", line=dict(color=goal_c, width=2.5))

    pitch_line(fig, [60, 120, 120, 60, 60], [0, 0, 80, 80, 0], line_c, 2.5)
    pitch_line(fig, [102, 120], [18, 18], line_c)
    pitch_line(fig, [102, 102], [18, 62], line_c)
    pitch_line(fig, [102, 120], [62, 62], line_c)
    pitch_line(fig, [114, 120], [30, 30], line_c)
    pitch_line(fig, [114, 114], [30, 50], line_c)
    pitch_line(fig, [114, 120], [50, 50], line_c)
    th = np.linspace(np.pi - 0.9273, np.pi + 0.9273, 40)
    pitch_line(fig, 108 + 10 * np.cos(th), 40 + 10 * np.sin(th), line_c)
    th = np.linspace(-np.pi / 2, np.pi / 2, 50)
    pitch_line(fig, 60 + 10 * np.cos(th), 40 + 10 * np.sin(th), line_c)
    for cy, sgn in ((0, 1), (80, -1)):  # cung góc sân
        th = np.linspace(0, np.pi / 2, 12)
        pitch_line(fig, 120 - 1.2 * np.cos(th), cy + sgn * 1.2 * np.sin(th), line_c)
    fig.add_trace(go.Scatter(x=[40], y=[108], mode="markers", marker=dict(size=6, color=line_c),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[36, 44], y=[120, 120], mode="markers", marker=dict(size=9, color="#fff" if dark else "#0f172a",
                             line=dict(color="#0f172a" if dark else "#fff", width=1.5)), hoverinfo="skip", showlegend=False))

    # phễu sút (bỏ qua khi góc quá hẹp, ví dụ đứng ở cờ góc)
    if float(goal_angle(x0, y0)) > 1.5:
        fig.add_trace(go.Scatter(
            x=[y0, 36, 44, y0], y=[x0, 120, 120, x0], mode="lines", fill="toself",
            fillcolor="rgba(255,226,89,.24)", line=dict(color="rgba(255,226,89,.85)", width=1.6, dash="dot"),
            hoverinfo="skip", showlegend=False))

    # bóng: vòng sáng + bóng
    fig.add_trace(go.Scatter(x=[y0], y=[x0], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=44, color="rgba(255,226,89,.20)")))
    fig.add_trace(go.Scatter(x=[y0], y=[x0], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=28, color="rgba(255,226,89,.35)")))
    fig.add_trace(go.Scatter(x=[y0], y=[x0], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=17, color="#ffe259", line=dict(color="#0f172a", width=3))))

    if interactive:  # lưới vô hình để bắt cú bấm
        gx, gy, custom = click_grid()
        fig.add_trace(go.Scatter(
            x=gy, y=gx, mode="markers", customdata=custom, showlegend=False,
            marker=dict(size=10, color="rgba(0,0,0,0)"),
            selected=dict(marker=dict(opacity=0)), unselected=dict(marker=dict(opacity=0)),
            hovertemplate=(f"<b>{T['hover_d']}</b>: %{{customdata[0]:.1f}} m<br>"
                           f"<b>{T['hover_a']}</b>: %{{customdata[1]:.1f}}°<extra></extra>"),
        ))

    fig.add_annotation(x=40, y=125.4, text=f"<b>{T['goal']}</b>", showarrow=False,
                       font=dict(size=11, color="rgba(148,163,192,1)"))
    fig.update_layout(
        height=470, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-6, 86], visible=False, fixedrange=True, constrain="domain"),
        yaxis=dict(range=[57, 127], visible=False, fixedrange=True, scaleanchor="x", scaleratio=1),
        dragmode=False, clickmode="event+select", hovermode="closest", hoverdistance=-1,
        hoverlabel=dict(bgcolor="rgba(15,23,42,.94)", font=dict(color="white", size=13), bordercolor="rgba(0,0,0,0)"),
    )
    return fig


# =====================================================================
# 6. DỰ ĐOÁN
# =====================================================================
def predict_xg(x, y, play_pattern, shot_type, body, technique, cone, close, pressure, first):
    row = pd.DataFrame(0.0, index=[0], columns=FEATURES)
    row["distance"] = float(goal_distance(x, y))
    row["angle"] = float(goal_angle(x, y))
    row["defenders_in_cone"] = cone
    row["close_defenders"] = close
    row["under_pressure"] = int(pressure)
    row["shot_first_time"] = int(first)
    # Các giá trị tham chiếu (From Corner, Free Kick, Head, Backheel) không có cột => toàn 0
    for prefix, val in [("play_pattern", play_pattern), ("shot_type", shot_type),
                        ("shot_body_part", body), ("shot_technique", technique)]:
        col = f"{prefix}_{val}"
        if col in row.columns:
            row[col] = 1
    return float(model.predict_proba(row)[0][1])


def verdict_level(p):
    return 0 if p < 0.06 else 1 if p < 0.15 else 2 if p < 0.30 else 3 if p < 0.60 else 4


# =====================================================================
# 7. DỰNG TRANG
# =====================================================================
SITS = ["open", "fk", "corner", "pen"]
BODY = {
    "open": ["Left Foot", "Right Foot", "Head", "Other"],
    "corner": ["Head", "Left Foot", "Right Foot"],
    "corner_direct": ["Left Foot", "Right Foot"],
    "fk": ["Left Foot", "Right Foot"],
    "pen": ["Left Foot", "Right Foot"],
}
# Giá trị (hậu vệ chắn, hậu vệ áp sát) dùng cho tình huống mà ô nhập bị ẩn.
# Phạt góc: vòng cấm luôn đông người nên để 0 sẽ làm xG bị thổi phồng.
# Gợi ý: thay bằng trung vị thật từ dữ liệu huấn luyện của bạn, ví dụ
#   df[df.play_pattern == "From Corner"][["defenders_in_cone", "close_defenders"]].median()
HIDDEN_DEFAULTS = {"corner": (2, 0)}
PATTERNS = ["Regular Play", "From Counter", "From Throw In", "From Keeper", "From Goal Kick", "Other"]


def tech_options(ctx, body):
    if ctx in ("fk", "pen", "corner_direct") or body == "Other":
        return ["Normal"]
    if body == "Head":
        return ["Normal", "Diving Header"]
    if ctx == "corner":
        return ["Normal", "Half Volley", "Volley"]
    return ["Normal", "Half Volley", "Volley", "Lob", "Backheel", "Overhead Kick"]


# --- thanh trên cùng: ngôn ngữ + chế độ sáng/tối
top = st.columns([5, 2.4, 1.8], vertical_alignment="center")
with top[0]:
    st.markdown("<div class='brand'>⚽ Messi xG Lab</div>", unsafe_allow_html=True)
with top[1]:
    st.radio("lang", ["vi", "en"], format_func=lambda v: "Tiếng Việt" if v == "vi" else "English",
             horizontal=True, key="pill_lang", label_visibility="collapsed")
T = TXT[st.session_state["pill_lang"]]
with top[2]:
    dark = st.toggle(T["dark"], value=True, key="dark")
inject_css(dark)

hero_slot = st.container()  # banner kết quả: được vẽ sau khi tính xong

# --- vị trí cú sút: đọc cú bấm gần nhất trên sân (trước khi dựng bất cứ thứ gì)
st.session_state.setdefault("pos", (105.0, 40.0))
clicked = st.session_state.get("pitch")
if clicked:
    try:
        pts = clicked["selection"]["points"]
    except (KeyError, TypeError):
        pts = []
    if pts:
        st.session_state["pos"] = (float(pts[-1]["y"]), float(pts[-1]["x"]))

left, right = st.columns([1.3, 1], gap="medium")

# --- cột phải: nhập liệu
with right:
    with st.container(key="card_inputs"):
        st.markdown(f"<div class='card-title'>{T['sit_title']}</div>", unsafe_allow_html=True)
        sit = st.radio("sit", SITS, format_func=lambda s: T["sit"][s], horizontal=True,
                       key="pill_situation", label_visibility="collapsed")
        st.markdown(f"<div class='hint'>{T['hint'][sit]}</div>", unsafe_allow_html=True)

        play_pattern = {"open": "Regular Play", "fk": "From Free Kick", "corner": "From Corner", "pen": "Regular Play"}[sit]
        shot_type = {"open": "Open Play", "fk": "Free Kick", "corner": "Open Play", "pen": "Penalty"}[sit]
        cone, close = HIDDEN_DEFAULTS.get(sit, (0, 0))
        pressure = first = False
        direct_corner, corner_side = False, "L"
        ctx = sit

        if sit == "corner":
            st.markdown(f"<div class='sec'>{T['corner_mode_title']}</div>", unsafe_allow_html=True)
            mode = st.radio("cm", ["after", "direct"], format_func=lambda v: T["corner_mode"][v], horizontal=True,
                            key="pill_corner_mode", label_visibility="collapsed")
            direct_corner = mode == "direct"
            if direct_corner:
                ctx = "corner_direct"
                corner_side = st.radio("cs", ["L", "R"], format_func=lambda v: T["side"][v], horizontal=True,
                                       key="pill_corner_side", label_visibility="collapsed")

        if sit == "open":
            st.markdown(f"<div class='sec'>{T['sec_ctx']}</div>", unsafe_allow_html=True)
            play_pattern = st.selectbox(T["pattern"], PATTERNS, format_func=lambda v: T["pat"][v], key="pattern_open")
            st.markdown(f"<div class='sec'>{T['sec_def']}</div>", unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            cone = d1.slider(T["cone"], 0, 11, 1, key="cone_open", help=T["cone_help"])
            close = d2.slider(T["close"], 0, 5, 0, key="close_open")
        elif sit == "fk":
            st.markdown(f"<div class='sec'>{T['sec_def']}</div>", unsafe_allow_html=True)
            cone = st.slider(T["wall"], 0, 8, 4, key="wall_fk")

        if sit == "open" or (sit == "corner" and not direct_corner):
            c1, c2 = st.columns(2)
            pressure = c1.checkbox(T["pressure"], key=f"pressure_{sit}")
            first = c2.checkbox(T["first"], key=f"first_{sit}")

        st.markdown(f"<div class='sec'>{T['sec_tech']}</div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        body = b1.selectbox(T["body"], BODY[ctx], format_func=lambda v: T["bod"][v], key=f"body_{ctx}")
        techs = tech_options(ctx, body)
        technique = "Normal"
        if len(techs) > 1:
            technique = b2.selectbox(T["tech"], techs, format_func=lambda v: T["tec"][v], key=f"tech_{ctx}_{body}")

# --- tính toán
fixed_pos = {"pen": (108.0, 40.0)}.get(sit)
if direct_corner:
    fixed_pos = (120.0, 0.0 if corner_side == "L" else 80.0)
x_pos, y_pos = fixed_pos or st.session_state["pos"]
xg = predict_xg(x_pos, y_pos, play_pattern, shot_type, body, technique, cone, close, pressure, first)
dist_m = float(goal_distance(x_pos, y_pos)) * YARD_TO_M
ang = float(goal_angle(x_pos, y_pos))
lvl = verdict_level(xg)
pct = xg * 100

with hero_slot:
    st.markdown(
        f"""
        <div class='hero'>
          <div>
            <span class='eyebrow'>{T['eyebrow']}</span>
            <div class='hero-title'>{T['hero_title']}</div>
            <div class='hero-sub'>{T['hero_sub']}</div>
          </div>
          <div class='glass'>
            <div class='res-label'>{T['xg']}</div>
            <div class='res-row'>
              <div class='res-value'>{pct:.1f}<small>%</small></div>
              <span class='pill l{lvl}'>{T['verdicts'][lvl]}</span>
            </div>
            <div class='bar'><div class='mask' style='left:{pct:.1f}%'></div></div>
            <div class='tiles'>
              <div class='tile'><div class='v'>{dist_m:.1f} m</div><div class='l'>{T['dist']}</div></div>
              <div class='tile'><div class='v'>{ang:.1f}°</div><div class='l'>{T['angle']}</div></div>
              <div class='tile'><div class='v'>≈ {round(pct)}</div><div class='l'>{T['per100']}</div></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- cột trái: sân bóng
with left:
    with st.container(key="card_pitch"):
        st.markdown(f"<div class='card-title'>{T['pitch_title']}</div>", unsafe_allow_html=True)
        sub = T["pen_note"] if sit == "pen" else T["direct_note"] if direct_corner else T["pitch_hint"]
        st.markdown(f"<div class='card-sub'>{sub}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<span class='chip'>📍 {T['pos']}: X {x_pos:.0f} · Y {y_pos:.0f}</span>"
            f"<span class='chip'>📏 {dist_m:.1f} m</span><span class='chip'>📐 {ang:.1f}°</span>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_pitch(dark, (x_pos, y_pos), interactive=fixed_pos is None, T=T),
            key="pitch", on_select="rerun", selection_mode="points",
            width="stretch", config={"displayModeBar": False, "scrollZoom": False},
        )

st.markdown(f"<div class='foot'>{T['foot']}</div>", unsafe_allow_html=True)
