"""import json
from transformers import pipeline
from tqdm import tqdm
import itertools

# Load NLI model
nli_model = pipeline('zero-shot-classification', model='roberta-large-mnli')

# Path to your JSON file
json_file_path = '/u/gayathri/csc2541/classified_entries_with_multiple_treatments_or_causes.json'  # Replace with the path to your JSON file

# Reading the questions and answers from the JSON file
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize a dictionary to hold the results
contradictions = {}

# Analyzing the entries for contradictions
for entry in tqdm(data, desc="Analyzing Entries"):
    treatments = entry.get('treatments', [])
    causes = entry.get('causes', [])

    # Check for contradictions in treatments
    for treatment1, treatment2 in itertools.combinations(treatments, 2):
        combined_sequence = f"According to some, {treatment1} However, another view is that {treatment2}"
        result = nli_model(
            sequences=combined_sequence,
            candidate_labels=["contradiction", "entailment", "neutral"]
        )
        if result['labels'][0] == 'contradiction':
            contradictions.setdefault(f"Entry {entry['entry']}", {}).setdefault('treatments', []).append({
                "treatment1": treatment1,
                "treatment2": treatment2,
                "result": result['labels'][0],
                "score": result['scores'][0]
            })

    # Check for contradictions in causes
    for cause1, cause2 in itertools.combinations(causes, 2):
        combined_sequence = f"According to some, {cause1} However, another view is that {cause2}"
        result = nli_model(
            sequences=combined_sequence,
            candidate_labels=["contradiction", "entailment", "neutral"]
        )
        if result['labels'][0] == 'contradiction':
            contradictions.setdefault(f"Entry {entry['entry']}", {}).setdefault('causes', []).append({
                "cause1": cause1,
                "cause2": cause2,
                "result": result['labels'][0],
                "score": result['scores'][0]
            })

# Path to save the results JSON file
results_json_path = '/u/gayathri/csc2541/classified_entries_results.json'  # Replace with your desired save path

# Writing the results to a JSON file
with open(results_json_path, 'w', encoding='utf-8') as f:
    json.dump(contradictions, f, ensure_ascii=False, indent=4)

print(f"Results saved to {results_json_path}") """
import json
from transformers import pipeline
from tqdm import tqdm
import itertools

import torch
print(torch.cuda.is_available())

# Load NLI model
nli_model = pipeline('zero-shot-classification', model='roberta-large-mnli')

# Path to your JSON file
json_file_path = '/u/gayathri/csc2541/classified_entries_with_multiple_treatments_or_causes.json'

# Reading the questions and answers from the JSON file
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize a dictionary to hold the results
contradictions = {}

# Statistics
total_pairs = 0
contradicting_pairs = 0

# Analyzing the entries for contradictions
for entry in tqdm(data, desc="Analyzing Entries"):
    treatments = entry.get('treatments', [])
    causes = entry.get('causes', [])

    # Analyzing treatments for contradictions
    for treatment1, treatment2 in itertools.combinations(treatments, 2):
        total_pairs += 1
        combined_sequence = f"Statement one: {treatment1}. Statement two: {treatment2}"
        result = nli_model(
            sequences=combined_sequence,
            candidate_labels=["contradiction", "entailment", "neutral"]
        )
        if result['labels'][0] == 'contradiction':
            contradicting_pairs += 1
            contradictions.setdefault(f"Entry {entry['entry']}", {}).setdefault('treatments', []).append({
                "treatment1": treatment1,
                "treatment2": treatment2,
                "result": result['labels'][0],
                "score": result['scores'][0]
            })

    # Analyzing causes for contradictions
    for cause1, cause2 in itertools.combinations(causes, 2):
        total_pairs += 1
        combined_sequence = f"Statement one: {cause1}. Statement two: {cause2}"
        result = nli_model(
            sequences=combined_sequence,
            candidate_labels=["contradiction", "entailment", "neutral"]
        )
        if result['labels'][0] == 'contradiction' and result['scores'][0] > 0.5:
            contradicting_pairs += 1
            contradictions.setdefault(f"Entry {entry['entry']}", {}).setdefault('causes', []).append({
                "cause1": cause1,
                "cause2": cause2,
                "result": result['labels'][0],
                "score": result['scores'][0]
            })

# Calculate statistics
percentage_of_contradictions = (contradicting_pairs / total_pairs) * 100 if total_pairs > 0 else 0

# Displaying statistics
print(f"Total number of pairs analyzed: {total_pairs}")
print(f"Number of contradicting pairs: {contradicting_pairs}")
print(f"Percentage of contradicting pairs: {percentage_of_contradictions:.2f}%")

# Path to save the results JSON file
results_json_path = '/u/gayathri/csc2541/classified_entries_results1.json'

# Writing the results to a JSON file
with open(results_json_path, 'w', encoding='utf-8') as f:
    json.dump(contradictions, f, ensure_ascii=False, indent=4)

print(f"Results saved to {results_json_path}")

