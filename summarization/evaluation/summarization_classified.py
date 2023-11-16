# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:34:01 2023

@author: fybdt
"""

import json

with open("classified.json", "r") as file:
    data = json.load(file)
    
    
from collections import defaultdict    

def get_category(data, print_content = 0):
  # Create a defaultdict to group data by category
  grouped_data = defaultdict(list)
  # Iterate through each data point and concatenate texts by category
  for answer_list in data['classified_answers']:
     for text, category in answer_list:
        grouped_data[category] += text + " "

  for category, text in grouped_data.items():
     grouped_data[category] = ''.join(grouped_data[category])
     
  if print_content:
     for category, text in grouped_data.items():
         print(f"[{category}]")
         print(text)
     
  return data['subject'], data['content'], grouped_data
 

# question, content, answers = get_category(data[258])
# print(question)
# print(content)
# print(answers)       
 
def count_words(text):
    words = text.split()
    return len(words)       
 
    
from transformers import BartTokenizer, BartForConditionalGeneration

#tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
#model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')


def Bart(input_text, tokenizer, model, print_summuary=0, rate=0.33):
  # Tokenize the input text
  input_text = str(input_text)
  inputs = tokenizer(input_text, return_tensors="pt")

  # Generate the summary
  summary_ids = model.generate(**inputs, max_length=count_words(input_text), 
                               min_length=int(rate*count_words(input_text)), 
                               length_penalty=2, num_beams=4, early_stopping=True)
  summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

  # Print the generated summary
  if print_summuary:
    print("Original Text:")
    print(input_text)
    print("\nGenerated Summary:")
    print(summary)
  
  return summary




import sentencepiece
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 tokenizer and model
#tokenizer = T5Tokenizer.from_pretrained('t5-base')
#model = T5ForConditionalGeneration.from_pretrained('t5-base')


def T5(input_text, tokenizer, model, print_summuary=0, rate=0.33):
  # Tokenize the input text
  input_text = str(input_text)
  inputs = tokenizer("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)

  # Generate the summary
  summary_ids = model.generate(**inputs, max_length=count_words(input_text), 
                               min_length=int(rate*count_words(input_text)), 
                               length_penalty=0, num_beams=4, early_stopping=True)
  summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

  # Print the generated summary
  if print_summuary:
    print("Original Text:")
    print(input_text)
    print("\nGenerated Summary:")
    print(summary)
  
  return summary


def produce_summary(data, n=5, method="Bart"):
    if method == "Bart":
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    if method == "T5":
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
    for i in range(n):
        question, content, grouped_data = get_category(data[i])
        print(f"Question {i}: {question}")
        print("Answer:")
        for category, text in grouped_data.items():
            print(f"[{category}]")
            if method == "Bart":
                summary = Bart(text, tokenizer, model, 1)
            elif method == "T5":
                summary = T5(text, tokenizer, model, 1)
            
produce_summary(data) 
               