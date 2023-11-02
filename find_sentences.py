import os
import random
from pathlib import Path
import pandas as pd
from preprocessing import split_answer, load_dataset, add_newline_to_elipsis

"""
Cause: 38
Experience: 120
Information: 439
Suggestion: 164
Treatment: 107

F1 scores
['suggestion', 'cause', 'information', 'treatment', 'experience']
[0.77777778 0.         0.83597884 0.55555556 0.84444444]
"""

    
def is_suggestion(sentence: str, subject: str, content: str):
    word_indicators = [
        "see",
        "doctor",
        "check",
        "visit",
    ]
    return any(word in sentence for word in word_indicators)

def is_cause(sentence: str, subject: str, content: str):
    word_indicators = ["cause", "reason"]
    why_indicators = ["why"]
    for word in why_indicators:
        if subject and word in subject.lower():
            return True
        if content and word in content.lower():
            return True

    for word in word_indicators:
        if word in sentence.lower():
            return True

    return False


type2func = {
    "suggestion": is_suggestion,
    "cause": is_cause,
}

def select_sentences(data: pd.DataFrame, sent_type: str, num_sentences: int, output_path: str):
    random.seed(2345)
    chosen_set = set()
    selected = []
    while len(selected) < num_sentences:
        d = random.choice(data)
        answer = random.choice(d["nbestanswers"])
        sentence = random.choice(split_answer(answer))
        subject = d["subject"]
        content = d["content"]
        if (d["id"], sentence) in chosen_set:
            continue
        if not type2func[sent_type](sentence, subject, content):
            continue
        chosen_set.add((d["id"], answer))
        row = {
            "subject": d["subject"],
            "content": d["content"],
            "answer": answer,
            "sentence": sentence,
            "cat": d["cat"],
            "maincat": d["maincat"],
            "subcat": d["subcat"],
        }
        selected.append(row)

    df = pd.DataFrame(selected)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    DATA_DIR = Path(os.getenv("CSC2541_DIR")) / "data"
    file_path = DATA_DIR / "yahoo_health.json"
    data = load_dataset(file_path)
    sent_type = "cause"
    num_sentences = 500
    output_path = DATA_DIR / f"yahoo_health_{sent_type}_{num_sentences}.csv"
    select_sentences(data, sent_type, num_sentences, output_path)


