import json
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

print(f"Results saved to {results_json_path}")
