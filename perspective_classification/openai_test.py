import os
import openai
import pandas as pd
from tqdm import tqdm
from string import Template
import yaml
from utils import MAIN_DIR, DATA_DIR
from openai import OpenAI

"""
## OPENAI GPT3.5 TURBO BEST FINETUNED MODEL
Cost per sentence: ~0.224628/254 = 0.0008843622

Accuracy: 0.941
['experience', 'treatment', 'suggestion', 'information', 'cause']
[[ 32   0   0   2   0]
 [  0  30   1   0   0]
 [  0   2  50   2   0]
 [  0   1   2 111   0]
 [  0   0   0   5  16]]
[0.96969697 0.9375     0.93457944 0.94871795 0.86486486]
Weighted F1: 0.940


## OPENAI TOKEN EFFICIENT MODEL
Cost per sentence: ~0.152901/254 = 0.00060197244

Accuracy: 0.921
['experience', 'treatment', 'suggestion', 'information', 'cause']
[[ 34   0   0   2   0]
 [  0  32   0   0   0]
 [  0   4  47   4   0]
 [  2   2   0 107   1]
 [  0   1   0   4  14]]
[0.94444444 0.90140845 0.92156863 0.93449782 0.82352941]
Weighted F1: 0.921

## BERT FINETUNING
Accuracy: 81.10%
['experience', 'treatment', 'suggestion', 'information', 'cause']
[[ 29   0   0   5   0]
 [  0  23   2   6   0]
 [  0   7  41   6   0]
 [  2   3   1 103   5]
 [  0   1   2   8  10]]
[0.89230769 0.70769231 0.82       0.85123967 0.55555556]
Weighted F1: 0.808

## OPENAI GPT3.5 TURBO ZERO SHOT
Accuracy: 0.638
['experience', 'treatment', 'suggestion', 'information', 'cause']
[[32  0  2  0  2]
 [ 0 20 11  1  0]
 [ 0  1 46  6  2]
 [ 5 12 18 50 27]
 [ 0  1  1  3 14]]
[0.87671233 0.60606061 0.69172932 0.58139535 0.4375    ]
Weighted F1: 0.639

## NLI DEFAULT HYPOTHESES
Accuracy: 0.437
['treatment', 'suggestion', 'experience', 'cause', 'information']
[[ 6 17  0  3  5]
 [ 1 24  0  2 27]
 [ 1  2  6  6 19]
 [ 1  6  0  9  5]
 [ 7 16  1 24 66]]
[0.25531915 0.40336134 0.29268293 0.27692308 0.55932203]
Weighted F1: 0.430

## NLI SET HYPOTHESES
Accuracy: 0.433
['experience', 'cause', 'suggestion', 'treatment', 'information']
[[ 1  3  1  3 26]
 [ 0  0  0  2 19]
 [ 0  3  7  5 39]
 [ 0  0  8 13 10]
 [ 4  5  4 12 89]]
[0.05128205 0.         0.18918919 0.39393939 0.5993266 ]
Weighted F1: 0.364

"""

OPENAI_DIR = MAIN_DIR / "openai"
CONFIG_DIR =  OPENAI_DIR / "config"
OUTPUT_DIR = OPENAI_DIR / "data"
API_KEY = os.getenv("OPENAI_API_KEY")
# Initialize the OpenAI API client
openai.api_key = API_KEY
client = OpenAI()
FILENAME = "openai_test_data_11_11_zero_shot.csv"
CONFIG_FILE = CONFIG_DIR / "zero_shot_context.yaml"
MODEL = "gpt-3.5-turbo"
BEST_MODEL = "ft:gpt-3.5-turbo-0613:personal::8JEyByZO"

class ConversationGenerator:
    def __init__(self, config_file: str) -> None:
        config = yaml.safe_load(open(config_file))
        self.name = config["name"]
        self.system_message = config["system_message"]
        self.user_message_template = Template(config["user_message_template"])
        self.perspective_map = config["perspective_map"]
        self.model = MODEL

    def chat_with_gpt(self, messages):
        retries = 0
        while retries < 3:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    timeout=10,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(e)
                retries += 1
        return None
    
    def create_conversation(self, subject: str, content: str, sentence: str):
        conversation = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_message_template.safe_substitute(subject=subject, content=content, sentence=sentence)},
        ]
        return conversation
    
    def create_conversation_with_perspective(self, subject: str, content: str, sentence: str, perspective: str):
        conversation = self.create_conversation(subject, content, sentence)
        conversation.append({"role": "assistant", "content": self.perspective_map[perspective]})
        return conversation

    def classify_answers(self, input_path: str, output_path: str):
        data = pd.read_csv(input_path)
        data = data[data["s_perspective_1"].notnull()]

        new_data = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Classifying..."):
            question = row["subject"]
            context = row["content"]
            
            sentence = row["sentence"]
            conversation = self.create_conversation(question, context, sentence)
            response = self.chat_with_gpt(conversation)
            row["openai_s_perspective"] = response

            new_data.append(row)

        new_data = pd.DataFrame(new_data)
        new_data.to_csv(output_path)

def evaluate_labels(input_file: str, cg: ConversationGenerator, map_labels=True):
    data = pd.read_csv(input_file)
    correct_s = 0
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Evaluating labels"):
        if row["s_perspective_1"].lower() in row["openai_s_perspective"].lower():
            correct_s += 1
    
    from sklearn.metrics import confusion_matrix, f1_score
    print(f"Accuracy: {correct_s/len(data):.3f}")
    if map_labels:
        data["s_perspective_1"] = data["s_perspective_1"].map(lambda x: cg.perspective_map[x])

    data["s_perspective_1"] = data["s_perspective_1"].map(lambda x: x if x != 'cause, suggestion' else "cause")
    labels = list(data["s_perspective_1"].unique())
    
    def convert_labels(x):
        for label in labels:
            if label.lower() in x.lower():
                return label
        return x
    data["openai_s_perspective"] = data["openai_s_perspective"].map(convert_labels)
    
    print(labels)
    print(confusion_matrix(data["s_perspective_1"], data["openai_s_perspective"], labels=labels))
    print(f1_score(data["s_perspective_1"], data["openai_s_perspective"], labels=labels, average=None))
    weighted_f1 = f1_score(data["s_perspective_1"], data["openai_s_perspective"], labels=labels, average="weighted")
    print(f"Weighted F1: {weighted_f1:.3f}")
    

if __name__ == "__main__":
    input_file = DATA_DIR / FILENAME
    output_file = DATA_DIR / FILENAME.replace(".csv", "_classified.csv")
    generator = ConversationGenerator(CONFIG_FILE)
    # generator.classify_answers(input_file, output_file)
    evaluate_labels(output_file, generator, map_labels=False)