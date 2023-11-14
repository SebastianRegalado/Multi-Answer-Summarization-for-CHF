import json
import streamlit as st
from annotated_text import annotated_text
from utils import DATA_DIR, merge_sentences
 
with open(DATA_DIR / "yahoo_health_medical_short_classified.json", encoding="utf-8") as f:
    data = json.load(f)

if 'counter' not in st.session_state: 
    st.session_state.counter = 0

label2color = {
    "cause": "#ff0000",
    "suggestion": "#3d85c6",
    "experience": "#8e7cc3",
    "treatment": "#bf9000",
    "information": "#8fce00"
}

def convert_answer(answer):
    return [(sentence, label, label2color.get(label, "#000000")) for sentence, label in answer]

def show_question():
    st.title("Multi Answer Summarization for CHF")
    index = st.session_state.counter
    question = data[index]
    subject = question["subject"]
    content = question["content"]
    classified_answers = question["classified_answers"]
    st.write(f"## Subject: {subject}")
    st.write(f"## Content: {content}")

    for i, answer in enumerate(classified_answers):
        st.write(f"Answer {i+1}")
        answer = merge_sentences(answer)
        answer = convert_answer(answer)
        annotated_text(*answer)

    st.session_state.counter += 1
    if st.session_state.counter >= len(questions):
        st.session_state.counter = 0

button = st.button("Next question", on_click=show_question)