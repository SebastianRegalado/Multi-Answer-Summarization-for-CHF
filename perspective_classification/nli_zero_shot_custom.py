from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import DATA_DIR, get_device

device = get_device()
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
nli_model.to(device)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

def predict_labels(data, hypotheses, classes):
    # pose sequence as a NLI premise and label as a hypothesis
    predictions = []
    perspectives = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        sentence = row["sentence"]
        s_perspective = row["s_perspective_1"]
        perspectives.append(s_perspective)

        best_cls = None
        best_prob = 0
        for hypothesis, cls in zip(hypotheses, classes):
            x = tokenizer.encode(sentence, hypothesis, return_tensors='pt',truncation="only_first")
            logits = nli_model(x.to(device))[0]
            entail_contradiction_logits = logits[:,[0,2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:,1].item()
            if prob_label_is_true > best_prob:
                best_prob = prob_label_is_true
                best_cls = cls

        predictions.append(best_cls)

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
    data = pd.read_csv(filepath)
    data = data[data["s_perspective_1"].notnull()]
    data = data[data["s_perspective_1"] != "odd"]
    hypothesis = [
        "This sentence is a suggestion.",
        "This sentence provides information.",
        "This mentions is a cause.",
        "This mentions a treatment.",
        "This describes a experience.",
    ]
    classes = [
        "suggestion",
        "information",
        "cause",
        "treatment",
        "experience",
    ]
    predict_labels(data, hypothesis, classes)

filepath = DATA_DIR / "openai_test_data_11_10_basic.csv"
predict(filepath)