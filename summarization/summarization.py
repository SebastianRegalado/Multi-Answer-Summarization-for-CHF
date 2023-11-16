# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:34:01 2023

@author: fybdt
"""

import json

with open("yahoo_health.json", "r") as file:
    data = json.load(file)
    

def get_text(input):
    question = input.get("subject", "")
    content = input.get("content", "")
    answers = input.get("nbestanswers", "")
    return question, content, answers

def count_words(text):
    words = text.split()
    return len(words)



question, content, answers = get_text(data[258])
print(question)
print(content)
print(answers)


from transformers import BartTokenizer, BartForConditionalGeneration

# Load pre-trained BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')


# Tokenize the input text
input_text = str(answers)
inputs = tokenizer(input_text, return_tensors="pt")

# Generate the summary
summary_ids = model.generate(**inputs, max_length=count_words(input_text), min_length=int(0.33*count_words(input_text)), length_penalty=2, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the generated summary
print("Question:")
print(question)
print("Original Answers:")
print(input_text)
print("\nGenerated Summary:")
print(summary)



import sentencepiece
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Tokenize the input text
input_text = "I have similar problems.When I wake up I have a stuffy nose but then in like an hour or two and I’m fine.I take Zyrtec every morning but before I go to bed I take a Benadryl and that seems to help.It may help if you wash your hair in the evening to get rid of any pollen that might be left in there."
inputs = tokenizer("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate the summary
summary_ids = model.generate(**inputs, max_length=count_words(input_text), min_length=int(0.33*count_words(input_text)), length_penalty=0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the generated summary
print("Original Text:")
print(input_text)
print("\nGenerated Summary:")
print(summary)



