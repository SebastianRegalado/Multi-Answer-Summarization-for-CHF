import re
import json
from nltk.tokenize import sent_tokenize
import typing as T

def find_punctuations_without_space(text: str):
    matches = []
    for m in re.finditer(r"(?i)[\.\?!,;:]+[a-z]+[^\.\?!,]", text):
        ### find position of last puntuation
        start, end = m.span()
        for i in range(end-1, start-1, -1):
            if text[i] in ".?!,;:":
                matches.append(i)
                break
    return matches

def find_urls_position(text: str):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    matches = []
    for m in re.finditer(regex, text):
        matches.append(m.span())
    return matches
    
def filter_punctuation_matches(punct_matches: T.List[int], url_matches: T.List[T.Tuple[int, int]]):
    valid_matches = []
    for p in punct_matches:
        is_valid = True
        for start, end in url_matches:
            if start <= p < end:
                is_valid = False
                break
        if is_valid:
            valid_matches.append(p)
    return valid_matches

def find_elispis(text: str):
    return [m.end()-2 for m in re.finditer(r"(?i)[\.\?!,;:]{2,}[a-z]", text)]

def add_newline_to_elipsis(text: str):
    elipsis_matches = find_elispis(text)
    if len(elipsis_matches) == 0:
        return text
    url_matches = find_urls_position(text)
    valid_matches = filter_punctuation_matches(elipsis_matches, url_matches)

    new_text = ""
    prev_end = 0
    for p in valid_matches:
        new_text += text[prev_end:p+1] + "\n"
        prev_end = p + 1
    new_text += text[prev_end:]
    return new_text

def add_spaces_to_punctuations(text: str):
    punct_matches = find_punctuations_without_space(text)
    if len(punct_matches) == 0:
        return text
    url_matches = find_urls_position(text)
    valid_matches = filter_punctuation_matches(punct_matches, url_matches)

    new_text = ""
    prev_end = 0
    for p in valid_matches:
        new_text += text[prev_end:p+1] + " "
        prev_end = p + 1
    new_text += text[prev_end:]
    return new_text

def split_answer(answer: str):
    answer = add_newline_to_elipsis(answer)
    answer = add_spaces_to_punctuations(answer)
    paragraphs = answer.split("\n")
    sentences = []
    for p in paragraphs:
        sentences.extend(sent_tokenize(p))
    sentences = [s for s in sentences if len(s) > 1 and not s.isspace()]
    if len(sentences) == 0:
        sentences = [answer]
    return sentences

def is_valid_answers(answers):
    if answers is None:
        return False
    num_answers = len(answers)
    if num_answers < 3 or num_answers > 7:
        return False
    return True

def load_dataset(file_path: str):
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    data = [d for d in data if is_valid_answers(d["nbestanswers"])]
    return data