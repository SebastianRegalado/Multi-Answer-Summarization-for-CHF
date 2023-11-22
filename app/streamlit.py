import json
import streamlit as st
from pathlib import Path
from annotated_text import annotated_text
from utils import DATA_DIR, merge_sentences
 
DATA_DIR = Path("report_data")

with open(DATA_DIR / "yahoo_health_medical_short_classified_contradiction.json", encoding="utf-8") as f:
    questions_data = json.load(f)

with open(DATA_DIR / "classified_summary_Bart100.json", encoding="utf-8") as f:
    summaries_data = json.load(f)

starting_subject = "A week after my tooth was pulled...?"

if 'counter' not in st.session_state:
    st.session_state.counter = 0
    for i, question in enumerate(questions_data):
        if question["subject"] == starting_subject:
            st.session_state.counter = i
            break

if 'display' not in st.session_state:
    st.session_state.display = "answers"

label2color = {
    "cause": "#FF9A8A",
    "suggestion": "#AED9E0",
    "experience": "#C3AED6",
    "treatment": "#FFF5BA",
    "information": "#BFD8AC"
}

st.markdown("""
            <style>
                div[data-testid="column"] {
                    width: fit-content !important;
                    flex: unset;
                }
                div[data-testid="column"] * {
                    width: fit-content !important;
                }
            </style>
            """, unsafe_allow_html=True)

          
def convert_answer(answer):
    return [(sentence, label, label2color.get(label, "#000000")) for sentence, label in answer]

def show_next_question():
    st.session_state.counter += 1
    if st.session_state.counter >= len(questions_data):
        st.session_state.counter = 0

def show_prev_question():
    st.session_state.counter -= 1
    if st.session_state.counter < 0:
        st.session_state.counter = len(questions_data) - 1

def show_question():
    index = st.session_state.counter
    question = questions_data[index]
    subject = question["subject"]
    content = question["content"]
    #display_heading("Subject", subject)
    st.markdown(f"<h3><b>Subject</b>: {subject}</h3>", unsafe_allow_html=True)
    display_heading("Content", content)
    st.divider()
    if st.session_state.display == "answers":
        show_answers()
    elif st.session_state.display == "summaries":
        show_summaries()
    elif st.session_state.display == "conflicting_pairs":
        show_conflicting_pairs()

def update_display(display):
    st.session_state.display = display

def display_heading(type: str, text: str):
    if text is None:
        return
    lines = text.split("\n")
    st.write(f"#### {type}: {lines[0]}")
    for line in lines[1:]:
        if line.strip() == "":
            continue
        st.write(f"#### {line}")

def show_answers():
    index = st.session_state.counter
    question = questions_data[index]
    classified_answers = question["classified_answers"]

    for i, answer in enumerate(classified_answers):
        st.write(f"##### Answer {i+1}")
        answer = merge_sentences(answer)
        answer = convert_answer(answer)
        annotated_text(*answer)
        if i < len(classified_answers) - 1:
            st.divider()

def show_summaries():
    index = st.session_state.counter
    question = summaries_data[index]
    summaries = question["summary"]
    for i, (label, summary) in enumerate(summaries.items()):
        st.write(f"##### {label.capitalize()} Summary")
        st.write(summary)
        if i < len(summaries) - 1:
            st.divider()

def show_conflicting_pairs():
    index = st.session_state.counter
    question = questions_data[index]
    for label in ["cause", "treatment"]:
        st.write(f"##### {label.capitalize()} conflicting pairs")
        contradicting_pairs = question[f"{label}_contradiction_pairs"]
        if len(contradicting_pairs) == 0:
            st.write("No conflicting pairs found.")
            st.divider()
        for pair in contradicting_pairs:
            sent1 = pair["cause1"]
            sent2 = pair["cause2"]
            st.write(f"###### {label.capitalize()} 1: {sent1}")
            st.write(f"###### {label.capitalize()} 2: {sent2}")
            st.divider()


st.title("Multi Answer Summarization for CHF")
col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
col1.button("Answers", on_click=lambda: update_display("answers"))
col2.button("Summaries", on_click=lambda: update_display("summaries"))
col3.button("Conflicting pairs", on_click=lambda: update_display("conflicting_pairs"))
# col4.button("Prev question", on_click=show_prev_question)
# col5.button("Next question", on_click=show_next_question)
show_question()