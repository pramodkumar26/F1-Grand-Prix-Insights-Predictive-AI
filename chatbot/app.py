import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
# f1.db and the mlruns artifacts are both looked up relative to the working directory
os.chdir(PROJECT_ROOT)

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
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 6rem;
    max-width: 1080px;
    margin: 0 auto;
}

html, body, [class*="css"], p, div, span, li {
    font-family: 'Titillium Web', -apple-system, sans-serif;
}

.eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #8A8A94;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Titillium Web', sans-serif;
    font-weight: 900;
    font-size: clamp(2.8rem, 6.5vw, 5.4rem);
    line-height: 0.92;
    letter-spacing: -2px;
    text-transform: uppercase;
    color: #F5F5F0;
    margin: 0 0 1.6rem 0;
}
.hero-title .accent {color: #E10600;}
.hero-lede {
    font-size: 1.02rem;
    font-weight: 300;
    line-height: 1.7;
    color: #A8A8B2;
    max-width: 32rem;
    margin-bottom: 2rem;
}
.rule {
    height: 1px;
    background: #2A2A33;
    margin: 2.6rem 0 1.8rem 0;
}
.stat-row {
    display: flex;
    flex-wrap: wrap;
    gap: 3.4rem;
}
.stat-num {
    font-weight: 700;
    font-size: 2rem;
    color: #F5F5F0;
    line-height: 1;
}
.panel {
    border: 1px solid #23232C;
    border-left: 2px solid #E10600;
    padding: 1.4rem 1.5rem;
    margin-top: 0.4rem;
}
.panel-title {
    font-size: 0.62rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #7A7A84;
    margin-bottom: 1rem;
}
.panel-item {
    font-size: 0.9rem;
    color: #C8C8D2;
    padding: 0.5rem 0;
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
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    padding: 0.95rem 2rem !important;
    transition: opacity 0.25s ease !important;
}
button[kind="primary"]:hover {opacity: 0.82 !important;}
button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #23232C !important;
    border-left: 2px solid #33333D !important;
    border-radius: 2px !important;
    color: #B8B8C2 !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 0.85rem 1rem !important;
    transition: border-left-color 0.25s ease, color 0.25s ease, background 0.25s ease !important;
}
button[kind="secondary"] p {text-align: left !important;}
button[kind="secondary"]:hover {
    border-left-color: #E10600 !important;
    color: #F5F5F0 !important;
    background: #101017 !important;
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
            tools.get_next_race, tools.get_championship_standings, tools.get_constructor_standings,
            tools.get_driver_season_results, tools.get_race_summary,
            tools.get_race_control_messages,
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
    st.markdown("""
    <div class="stat-row">
        <div><div class="stat-num">9</div><div class="stat-label">Seasons</div></div>
        <div><div class="stat-num">184</div><div class="stat-label">Races</div></div>
        <div><div class="stat-num">5</div><div class="stat-label">Trained models</div></div>
        <div><div class="stat-num">201K</div><div class="stat-label">Laps analysed</div></div>
    </div>
    """, unsafe_allow_html=True)


def reset_conversation():
    st.session_state.view = 'landing'
    st.session_state.display_history = []
    st.session_state.current_suggestions = []
    st.session_state.pending_prompt = None
    get_chat.clear()


def render_chat():
    left, right = st.columns([3, 1], vertical_alignment='center')
    with left:
        if st.button('Grand Prix Insights', type='tertiary', key='go_home'):
            reset_conversation()
            st.rerun()
    with right:
        st.markdown(
            f'<div class="chat-meta">{answer_policy.MIN_YEAR}&ndash;{answer_policy.MAX_YEAR}</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="head-rule"></div>', unsafe_allow_html=True)

    registry.load_models()
    registry.load_explainers()

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
