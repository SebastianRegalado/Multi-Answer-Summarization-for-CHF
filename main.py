import csv
import itertools
import random
from transformers import pipeline

# Load NLI model
nli_model = pipeline('zero-shot-classification', model='roberta-large-mnli')

# Path to your CSV file
csv_file_path = '/content/drive/MyDrive/UofT/ML-healthcare/project/yahoo_health/filtered_qa_pairs.csv'

# Reading the questions and answers from the CSV file
questions_and_answers = {}
with open(csv_file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        question, answer = row
        if question not in questions_and_answers:
            questions_and_answers[question] = []
        questions_and_answers[question].append(answer)

# Maximum number of answers to sample for comparison
max_sample_size = 6  # Adjust as needed

# Check for contradictions within each set of answers
for question, answers in questions_and_answers.items():
    if len(answers) > 1:
        # Randomly sample answers if there are too many
        sampled_answers = random.sample(answers, min(len(answers), max_sample_size))

        for answer1, answer2 in itertools.combinations(sampled_answers, 2):
            # Adding context to the sequence
            combined_sequence = f"According to some, {answer1} However, another view is that {answer2}"

            # Run the model
            result = nli_model(
                sequences=combined_sequence,
                candidate_labels=["contradiction", "entailment", "neutral"]
            )

            label = result['labels'][0]
            score = result['scores'][0]

            if label == 'contradiction':
                print(f"\nContradiction detected in responses to: '{question}'")
                print(f"Comparing: '{answer1}' and '{answer2}'")
                print(f"Result: {label} (Score: {score:.2f})")
                break  # Stop after finding the first contradiction for this question