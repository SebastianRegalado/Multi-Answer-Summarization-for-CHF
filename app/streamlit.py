import json
from streamlit import st
from annotated_text import annotated_text
from utils import DATA_DIR
 
with open(DATA_DIR / "yahoo_health_medical_short_classified.json", encoding="utf-8") as f:
    data = json.load(f)
questions = [q for q in data if "classified_answers" in q]

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
    index = st.session_state.counter
    question = questions[index]
    subject = question["subject"]
    content = question["content"]
    classified_answers = question["classified_answers"]
    st.write(f"Subject: {subject}")
    st.write(f"Content: {content}")
    st.write("Answers:")

    for answer in classified_answers:
        answer = convert_answer(answer)
        annotated_text(*answer)

    st.session_state.counter += 1
    if st.session_state.counter >= len(questions):
        st.session_state.counter = 0

st.title("Multi Answer Summarization for CHF")
button = st.button("Next question", on_click=show_question)
