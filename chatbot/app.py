import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
# f1.db and the mlruns artifacts are both looked up relative to the working directory
os.chdir(PROJECT_ROOT)

import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from google.genai import errors

from chatbot import tools, answer_policy, registry

GEMINI_MODEL = 'gemini-3.5-flash-lite'

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&display=swap');

#MainMenu, footer, header {visibility: hidden;}
/* streamlit appends anchor links to markdown headings */
[data-testid="stHeaderActionElements"], .hero-title a {display: none !important;}
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 3rem;
    max-width: 1080px;
    margin: 0 auto;
}

html, body, [class*="css"], p, div, span, li {
    font-family: 'Titillium Web', -apple-system, sans-serif;
}

.eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #8A8A94;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'Titillium Web', sans-serif;
    font-weight: 900;
    font-size: clamp(2.4rem, 5vw, 4.1rem);
    line-height: 0.9;
    letter-spacing: -1.5px;
    text-transform: uppercase;
    color: #F5F5F0;
    margin: 0 0 1.1rem 0;
}
.hero-title .accent {color: #E10600;}
.hero-lede {
    font-size: 0.95rem;
    font-weight: 300;
    line-height: 1.6;
    color: #A8A8B2;
    max-width: 32rem;
    margin-bottom: 1.3rem;
}
.rule {
    height: 1px;
    background: #2A2A33;
    margin: 1.5rem 0 1.1rem 0;
}
.stat-row {
    display: flex;
    flex-wrap: wrap;
    gap: 3.2rem;
}
.stat-num {
    font-weight: 700;
    font-size: 1.7rem;
    color: #F5F5F0;
    line-height: 1;
}
.panel {
    border: 1px solid #23232C;
    border-left: 2px solid #E10600;
    padding: 1rem 1.2rem;
    margin-top: 0.2rem;
}
.panel-title {
    font-size: 0.6rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #7A7A84;
    margin-bottom: 0.7rem;
}
.panel-item {
    font-size: 0.85rem;
    color: #C8C8D2;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1A1A22;
}
.panel-item:last-child {border-bottom: none;}
.panel-item b {color: #F5F5F0; font-weight: 600;}
.stat-label {
    font-size: 0.62rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #7A7A84;
    margin-top: 0.5rem;
}
button[kind="tertiary"] {
    font-weight: 900 !important;
    font-size: 1.15rem !important;
    letter-spacing: -0.5px !important;
    text-transform: uppercase !important;
    color: #F5F5F0 !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    transition: opacity 0.25s ease !important;
}
button[kind="tertiary"]::after {content: "."; color: #E10600;}
button[kind="tertiary"]:hover {opacity: 0.6 !important;}
.head-rule {
    height: 1px;
    background: #2A2A33;
    margin: 0.7rem 0 0.4rem 0;
}
.chat-meta {
    font-size: 0.62rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #7A7A84;
    padding-top: 0.55rem;
    text-align: right;
}

[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    border-bottom: 1px solid #1E1E26;
    border-radius: 0;
    padding: 1.4rem 0;
    animation: rise 0.5s ease both;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {font-size: 0.97rem; line-height: 1.75; max-width: 46rem;}
@keyframes rise {
    from {opacity: 0; transform: translateY(8px);}
    to   {opacity: 1; transform: translateY(0);}
}

button[kind="primary"] {
    background: #E10600 !important;
    border: none !important;
    border-radius: 2px !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.8rem !important;
    transition: opacity 0.25s ease !important;
}
button[kind="primary"]:hover {opacity: 0.82 !important;}
button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #23232C !important;
    border-left: 2px solid #33333D !important;
    border-radius: 2px !important;
    color: #B8B8C2 !important;
    font-size: 0.8rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 0.6rem 0.9rem !important;
    transition: border-left-color 0.25s ease, color 0.25s ease, background 0.25s ease !important;
}
button[kind="secondary"] p {text-align: left !important;}
button[kind="secondary"]:hover {
    border-left-color: #E10600 !important;
    color: #F5F5F0 !important;
    background: #101017 !important;
}

.next-row {
    display: flex;
    align-items: center;
    gap: 2.4rem;
    margin: 2.2rem 0 2.4rem 0;
}
.track-bg {
    flex: 0 0 auto;
    width: 165px;
    height: 165px;
    pointer-events: none;
}
.track-bg path {
    fill: none;
    stroke: #E10600;
}
.track-line {
    stroke-width: 2.2;
    opacity: 0.35;
}
.track-dash {
    stroke-width: 1.4;
    stroke-dasharray: 3 9;
    opacity: 0.75;
    animation: crawl 9s linear infinite;
}
@keyframes crawl {
    to {stroke-dashoffset: -120;}
}
.track-car {
    fill: #FF2A22;
    filter: drop-shadow(0 0 3px rgba(225, 6, 0, 0.9));
}
.next-card {
    border-left: 2px solid #E10600;
    padding: 0.2rem 0 0.2rem 1.2rem;
}
.next-label {
    font-size: 0.6rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #7A7A84;
    margin-bottom: 0.5rem;
}
.next-race {
    font-weight: 700;
    font-size: 1.55rem;
    color: #F5F5F0;
    line-height: 1.15;
    letter-spacing: -0.5px;
}
.next-meta {
    font-size: 0.85rem;
    color: #9A9AA5;
    margin-top: 0.35rem;
}
.suggest-label {
    font-size: 0.62rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #7A7A84;
    margin: 1.6rem 0 0.7rem 0;
}
.f1-loading {
    position: relative;
    height: 30px;
    overflow: hidden;
    border-top: 1px solid #2A2A33;
    border-bottom: 1px solid #2A2A33;
}
.f1-car {
    position: absolute;
    top: 50%;
    transform: translateY(-50%) scaleX(-1);
    font-size: 17px;
    animation: drive 1.5s linear infinite;
}
@keyframes drive {
    0%   {left: -26px;}
    100% {left: 100%;}
}
[data-testid="stChatInput"] textarea {font-size: 0.95rem;}
</style>
"""

LOADING_HTML = '<div class="f1-loading"><span class="f1-car">🏎️</span></div>'

SUGGESTION_PROMPT = """Based only on this exchange, suggest exactly 3 short, natural follow-up
questions a user might ask next about F1 data (drivers, races, standings, seasons {min_year}-{max_year}).
One per line, no numbering, no quotes, no extra text.

User asked: "{question}"
Assistant answered: "{answer}"
"""

st.set_page_config(page_title='Grand Prix Insights', page_icon='🏎️', layout='wide')
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets['GEMINI_API_KEY'])


@st.cache_resource
def get_chat():
    client = get_client()
    config = types.GenerateContentConfig(
        system_instruction=answer_policy.SYSTEM_PROMPT,
        tools=[
            tools.predict_podium, tools.predict_win, tools.predict_points,
            tools.explain_strategy_shift, tools.estimate_overtakes,
            tools.get_teammate_scorecard, tools.get_pit_stop_scorecard, tools.get_sprint_result,
            tools.get_race_results, tools.get_qualifying_results, tools.get_sprint_qualifying,
            tools.predict_upcoming_race,
            tools.predict_from_hypothetical_grid,
            tools.get_next_race, tools.get_championship_standings, tools.get_constructor_standings,
            tools.get_driver_season_results, tools.get_race_summary,
            tools.get_race_control_messages, tools.get_season_calendar, tools.get_driver_career_stats,
        ],
    )
    return client.chats.create(model=GEMINI_MODEL, config=config)


def get_suggestions(question, answer):
    try:
        client = get_client()
        prompt = SUGGESTION_PROMPT.format(
            min_year=answer_policy.MIN_YEAR, max_year=answer_policy.MAX_YEAR,
            question=question, answer=answer,
        )
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        lines = [l.strip('-*0123456789. ').strip() for l in resp.text.strip().split('\n') if l.strip()]
        return lines[:3]
    except Exception:
        return []


EXAMPLE_QUESTIONS = [
    "Who's leading the championship?",
    'When is the next race?',
    "Verstappen's podium chance at 2024 Miami",
    'Which team had the best pit stops in 2025?',
    'How did Hamilton compare to his teammate in 2024?',
    "Why did Norris gain places at 2024 Monza?",
]


def start_with(question=None):
    st.session_state.view = 'chat'
    if question:
        st.session_state.pending_prompt = question
    st.rerun()


@st.cache_data(ttl=600)
def dataset_stats():
    """Read the headline numbers from the database rather than hardcoding them, so they
    grow on their own as each race weekend is pulled in."""
    conn = sqlite3.connect('f1.db')
    try:
        seasons = conn.execute('SELECT COUNT(DISTINCT year) FROM dim_race').fetchone()[0]
        races = conn.execute('SELECT COUNT(DISTINCT race_id) FROM fact_race_results').fetchone()[0]
        laps = conn.execute('SELECT COUNT(*) FROM fact_laps').fetchone()[0]
    finally:
        conn.close()
    laps_label = f'{laps / 1000:.0f}K' if laps < 1_000_000 else f'{laps / 1_000_000:.1f}M'
    return seasons, races, laps_label


def render_landing():
    hero, side = st.columns([3, 2], gap='large')

    with hero:
        st.markdown('<div class="eyebrow">Formula 1</div>', unsafe_allow_html=True)
        st.markdown(
            '<h1 class="hero-title">Grand Prix<br>Insights<span class="accent">.</span></h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hero-lede">Ask about any race, driver or season between '
            f'{answer_policy.MIN_YEAR} and {answer_policy.MAX_YEAR}. Every answer comes from trained '
            'models and recorded results, never guesswork.</p>',
            unsafe_allow_html=True,
        )
        if st.button('Start a conversation', type='primary', key='enter_chat'):
            start_with()

    with side:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">What it answers</div>
            <div class="panel-item"><b>Predictions</b> &nbsp;podium, win and points chances</div>
            <div class="panel-item"><b>Explanations</b> &nbsp;why a driver gained or lost places</div>
            <div class="panel-item"><b>Records</b> &nbsp;standings, results, sprints, pit stops</div>
            <div class="panel-item"><b>Head to head</b> &nbsp;teammate comparisons by season</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Try asking</div>', unsafe_allow_html=True)

    for row in range(0, len(EXAMPLE_QUESTIONS), 3):
        cols = st.columns(3, gap='small')
        for col, question in zip(cols, EXAMPLE_QUESTIONS[row:row + 3]):
            if col.button(question, key=f'ex_{question}', width='stretch'):
                start_with(question)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    seasons, races, laps = dataset_stats()
    st.markdown(
        '<div class="stat-row">'
        f'<div><div class="stat-num">{seasons}</div><div class="stat-label">Seasons</div></div>'
        f'<div><div class="stat-num">{races}</div><div class="stat-label">Races</div></div>'
        '<div><div class="stat-num">5</div><div class="stat-label">Trained models</div></div>'
        f'<div><div class="stat-num">{laps}</div><div class="stat-label">Laps analysed</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )


CHAT_STARTERS = [
    "Who's leading the championship?",
    'How did Verstappen do last season?',
    'Which team had the best pit stops in 2025?',
]


@st.cache_data
def next_race_circuit_path():
    """The real circuit outline for the upcoming race, traced from lap telemetry."""
    conn = sqlite3.connect(tools.DB_PATH)
    try:
        # must be the outline for THAT race's circuit, never the next circuit that happens to have one
        row = pd.read_sql("""
            SELECT co.svg_path
            FROM (
                SELECT circuit_id FROM dim_race
                WHERE race_date >= date('now')
                ORDER BY race_date ASC LIMIT 1
            ) next_race
            LEFT JOIN circuit_outline co ON next_race.circuit_id = co.circuit_id
        """, conn)
        if len(row) == 0 or pd.isna(row.svg_path.iloc[0]):
            return None
        return row.svg_path.iloc[0]
    except Exception:
        return None
    finally:
        conn.close()


def track_svg():
    path = next_race_circuit_path()
    if not path:
        return ''
    return (
        '<svg class="track-bg" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">'
        f'<path class="track-line" d="{path}"/>'
        f'<path class="track-dash" d="{path}"/>'
        '<circle class="track-car" r="2.1">'
        f'<animateMotion dur="7s" repeatCount="indefinite" rotate="auto" path="{path}"/>'
        '</circle></svg>'
    )


def render_empty_state():
    race = tools.get_next_race()
    if race.get('found'):
        try:
            when = datetime.strptime(race['race_date'][:10], '%Y-%m-%d').strftime('%d %B %Y')
        except (ValueError, TypeError):
            when = race['race_date']
        st.markdown(
            '<div class="next-row">'
            '<div class="next-card">'
            f'<div class="next-label">Next race &middot; Round {race["round"]}</div>'
            f'<div class="next-race">{race["race_name"]}</div>'
            f'<div class="next-meta">{race["circuit_name"]}, {race["country"]}'
            f' &nbsp;&middot;&nbsp; {when}</div>'
            '</div>'
            f'{track_svg()}'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="suggest-label">Start with</div>', unsafe_allow_html=True)
    cols = st.columns(len(CHAT_STARTERS), gap='small')
    for col, starter in zip(cols, CHAT_STARTERS):
        if col.button(starter, key=f'st_{starter}', width='stretch'):
            st.session_state.pending_prompt = starter
            st.rerun()


def reset_conversation():
    st.session_state.view = 'landing'
    st.session_state.display_history = []
    st.session_state.current_suggestions = []
    st.session_state.pending_prompt = None
    get_chat.clear()


def render_chat():
    if st.button('Grand Prix Insights', type='tertiary', key='go_home'):
        reset_conversation()
        st.rerun()
    st.markdown('<div class="head-rule"></div>', unsafe_allow_html=True)

    registry.load_models()
    registry.load_explainers()

    if not st.session_state.display_history:
        render_empty_state()

    for msg in st.session_state.display_history:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    if st.session_state.current_suggestions:
        st.markdown('<div class="suggest-label">You might also ask</div>', unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.current_suggestions))
        for i, suggestion in enumerate(st.session_state.current_suggestions):
            if cols[i].button(suggestion, key=f'sg_{i}_{hash(suggestion)}'):
                st.session_state.pending_prompt = suggestion

    prompt = st.chat_input('Ask about a race, driver, or season...')
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt:
        st.session_state.current_suggestions = []
        st.session_state.display_history.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.write(prompt)

        chat = get_chat()
        with st.chat_message('assistant'):
            placeholder = st.empty()
            placeholder.markdown(LOADING_HTML, unsafe_allow_html=True)
            succeeded = False
            try:
                response = chat.send_message(prompt)
                answer = response.text
                succeeded = True
            except errors.ClientError as e:
                if getattr(e, 'status', None) == 'RESOURCE_EXHAUSTED':
                    answer = "I've hit the free-tier rate limit for now, please wait a moment and try again."
                else:
                    answer = f'Something went wrong talking to the model: {e}'
            placeholder.write(answer)

        st.session_state.display_history.append({'role': 'assistant', 'content': answer})
        st.session_state.current_suggestions = get_suggestions(prompt, answer) if succeeded else []
        st.rerun()


if 'view' not in st.session_state:
    st.session_state.view = 'landing'
if 'display_history' not in st.session_state:
    st.session_state.display_history = []
if 'current_suggestions' not in st.session_state:
    st.session_state.current_suggestions = []
if 'pending_prompt' not in st.session_state:
    st.session_state.pending_prompt = None

if st.session_state.view == 'landing':
    render_landing()
else:
    render_chat()
