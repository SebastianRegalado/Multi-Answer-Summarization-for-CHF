import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import DATA_DIR, get_device

device = get_device()

nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
nli_model.to(device)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

MAX_LEN = 64
LABELS = ["information", "cause", "treatment", "suggestion", "experience", "clarification"]
pipe = pipeline(model="facebook/bart-large-mnli", device=device)

def is_valid_answers(answers):
    if answers is None:
        return False
    num_answers = len(answers)
    if num_answers < 3 or num_answers > 7:
        return False
    return True
 
class MyDataset(Dataset):
    def __init__(self, filepath: str):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        
        self.sentences = []
        for d in data:
            answers = d["nbestanswers"]
            if not is_valid_answers(answers):
                continue
            self.sentences.extend(answers)
        self.sentences = self.sentences[:MAX_LEN]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]

filepath = "data/yahoo_health.json"
dataset = MyDataset(filepath)

def get_labels_defaul_hypothesis():
    data = []
    batch_size = 32
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size, candidate_labels = LABELS), total=len(dataset)):
        scores = {
            label: score for label, score in zip(out["labels"], out["scores"])
        }
        sentence_info = {
            "sentence": out["sequence"],
            "pred_label": out["labels"][0],
            "scores": scores
        }
        data.append(sentence_info)

    with open("data/perspective_prediction.json", "w") as f:
        json.dump(data, f, indent=4)

class MyDataset2(Dataset):
    def __init__(self, filepath: str):
        data = pd.read_csv(filepath)
        data = data[data["s_perspective_1"].notnull()]
        data = data[data["s_perspective_1"] != "odd"]
        self.sentences = data["sentence"].tolist()
        self.answers = data["answer"].tolist()
        self.s_perspectives = data["s_perspective_1"].tolist()
        self.a_perspectives = data["a_perspective_1"].tolist()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]
    

def predict_labels(dataset_s, classes):
    # pose sequence as a NLI premise and label as a hypothesis
    data = []
    batch_size = 32
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset_s, batch_size=batch_size, candidate_labels = classes), total=len(dataset_s)):
        scores = {
            label: score for label, score in zip(out["labels"], out["scores"])
        }
        sentence_info = {
            "sentence": out["sequence"],
            "s_pred_label": out["labels"][0],
            "scores": scores,

        }
        data.append(sentence_info)

    perspectives = []
    predictions = []
    for i in range(len(dataset_s)):
        s_perspective = dataset_s.s_perspectives[i]
        perspectives.append(s_perspective)
        predictions.append(data[i]["s_pred_label"])


    from sklearn.metrics import confusion_matrix, f1_score
    acc = sum([1 if p == s else 0 for p, s in zip(predictions, perspectives)]) / len(predictions)
    print(f"Accuracy: {acc:.3f}")
    labels = list(set(perspectives))
    print(labels)
    print(confusion_matrix(perspectives, predictions, labels=labels))
    print(f1_score(perspectives, predictions, labels=labels, average=None))
    weighted_f1 = f1_score(perspectives, predictions, labels=labels, average="weighted")
    print(f"Weighted F1: {weighted_f1:.3f}")


def predict(filepath: str):
    dataset_s = MyDataset2(filepath)
    classes = [
        "suggestion",
        "information",
        "cause",
        "treatment",
        "experience",
    ]
    predict_labels(dataset_s, classes)

filepath = DATA_DIR / "openai_test_data_11_10_basic.csv"
predict(filepath)