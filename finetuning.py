import os
import typing as T
from pathlib import Path
from string import Template
import pandas as pd
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss

MAIN_DIR = Path(os.getenv("CSC2541_DIR", "."))
DATA_DIR = MAIN_DIR / "data"
EXP_DIR = MAIN_DIR / "experiments"
TEXT_COL = "sentence"
LABEL_COL = "s_perspective_1"
LABEL_MAP = {
    'suggestion': 0,
    'cause': 1,
    'information': 2, 
    'treatment': 3, 
    'experience': 4,
    'odd': 5,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

TEMPLATE_MAP = {
    "full": Template("Sentence: $sentence\nSubject: $subject\nContent: $content\nAnswer: $answer"),
    "context": Template("Sentence: $sentence\nSubject: $subject\nContent: $content"),
    "subject_sentence": Template("Sentence: $sentence\nSubject: $subject"),
    "sentence_only": Template("$sentence"),
}
TEMPLATE_NAMES = list(TEMPLATE_MAP.keys())

MODEL_NAME = "bert-base-uncased"
CLASS_WEIGHTS = None

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = CrossEntropyLoss(weight=CLASS_WEIGHTS)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
class Data:
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.train_data = train_data
        self.test_data = test_data
        self.train_dataset = load_dataset(train_data, tokenizer)
        self.test_dataset = load_dataset(test_data, tokenizer)

def create_context(data: pd.DataFrame, label_col: str, template: Template):
    subject = data["subject"]
    content = str(data["content"])
    answer = data["answer"]
    sentence = data[label_col]
    context = template.safe_substitute(subject=subject, content=content, answer=answer, sentence=sentence)
    return context

def tokenize_generator(tokenizer: AutoTokenizer):
    def tokenize_batch(batch):
        return tokenizer(batch[TEXT_COL], padding=True, truncation=True, return_tensors="pt")
    return tokenize_batch

def load_dataset(data: pd.DataFrame, tokenizer: AutoTokenizer) -> Dataset:
    tokenize_batch = tokenize_generator(tokenizer)
    dataset = Dataset.from_pandas(data, preserve_index=False)
    dataset = dataset.map(tokenize_batch, batched=True)
    return dataset

def load_data(file_path: Path, template: Template, remove_labels: T.List[str] = []):
    data = pd.read_csv(file_path)
    data[TEXT_COL] = data.apply(lambda row: create_context(row, TEXT_COL, template), axis=1)
    data = data[[TEXT_COL, LABEL_COL]]
    data = data[data[LABEL_COL].notna()]
    for label in remove_labels:
        data = data[data[LABEL_COL] != label]
    data = data.rename(columns={LABEL_COL: "label"})

    ### one hot encoding of labels
    labels = data["label"].unique()
    labels = {label: i for i, label in enumerate(labels)}
    data["label"] = data["label"].map(LABEL_MAP)

    y = data["label"]
    class_weights=class_weight.compute_class_weight('balanced',classes=np.unique(y),y=y)
    global CLASS_WEIGHTS
    CLASS_WEIGHTS=torch.tensor(class_weights,dtype=torch.float).to("mps")

    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return Data(train_data, test_data)

def finetune(
    exp_dir: Path, data: Data, remove_labels: T.List[str] = [], num_epochs: int = 20
):

    # Initialize the tokenizer and model
    num_labels = len(LABEL_MAP) - len(remove_labels)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # Freeze the model's base layers
    for name, param in model.named_parameters():
        if name.startswith("bert.embeddings") or name.startswith("bert.encoder"):
            param.requires_grad = False

    # #  Print the names of the model's parameters
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    

    # Define training arguments
    training_args = TrainingArguments(
        output_dir= exp_dir,
        evaluation_strategy="steps",
        eval_steps=50,
        save_total_limit=2,
        save_steps=50,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-3,
        weight_decay=0.01,
        logging_dir= exp_dir / "logs",
        logging_steps=50,
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=data.train_dataset,
        eval_dataset=data.test_dataset,
        compute_metrics=lambda p: {
            "accuracy": (p.predictions.argmax(1) == p.label_ids).mean(),
        },
    )

    # Fine-tune the model
    trainer.train()

    # # Evaluate the model
    results = trainer.evaluate()
    print(results)

    # Save the trained model
    model.save_pretrained(exp_dir / "model")

def eval_data(exp_dir: Path, data: Data):
    # Load the trained model
    model_dir = exp_dir / "model"
    output_file = exp_dir / "test_dataset_labeled.csv"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(exp_dir),
        train_dataset=data.train_dataset,
        eval_dataset=data.test_dataset,
        compute_metrics=lambda p: {
            "accuracy": (p.predictions.argmax(1) == p.label_ids).mean(),
        },
    )
    results = trainer.predict(data.test_dataset)
    predictions = results.predictions.argmax(-1)
    data.test_data["predictions"] = predictions
    data.test_data["predictions"] = data.test_data["predictions"].map(INV_LABEL_MAP)
    data.test_data["label"] = data.test_data["label"].map(INV_LABEL_MAP)
    data.test_data.to_csv(output_file, index=False)

def print_acc_metrics(exp_dir: Path, remove_labels: T.List[str] = []):
    labels = [l for l in list(LABEL_MAP.keys()) if l not in remove_labels]
    prediction_data = pd.read_csv(exp_dir / "test_dataset_labeled.csv")
    acc = (prediction_data["label"] == prediction_data["predictions"]).mean()

    # Print accuracy as percentage
    print(f"Accuracy: {acc * 100:.2f}%")
    print(labels)
    print(confusion_matrix(prediction_data["label"], prediction_data["predictions"], labels=labels))
    print(f1_score(prediction_data["label"], prediction_data["predictions"], labels=labels, average=None))

class ExperimentConfig:
    def __init__(self, template_name: str, remove_labels: T.List[str] = []):
        self.template_name = template_name
        self.template = TEMPLATE_MAP[template_name]
        self.remove_labels = remove_labels
        self.exp_name = self.create_exp_name()

    def create_exp_name(self):
        exp_name = "finetune_"
        exp_name += "wo_" if self.remove_labels else ""
        exp_name += "_".join(self.remove_labels)
        exp_name += "_" if self.remove_labels else ""
        exp_name += self.template_name
        return exp_name

def run_experiments(experiments: T.List[ExperimentConfig], num_epochs: int = 20):
    for exp in experiments:
        exp_dir = EXP_DIR / exp.exp_name
        data_file = DATA_DIR / "yahoo_health_labeled_10_27.csv"
        data = load_data(data_file, exp.template, exp.remove_labels)
        finetune(exp_dir, data, exp.remove_labels, num_epochs)
        eval_data(exp_dir, data)

if __name__ == "__main__":
    remove_label_options = [
        ["odd"]
    ]
    templates = ["sentence_only"]
    experiments = [
        ExperimentConfig(template_name, remove_labels)
        for remove_labels in remove_label_options
        for template_name in templates
    ]
    run_experiments(experiments, num_epochs=25)
    for exp in experiments:
        print(exp.exp_name)
        exp_dir = MAIN_DIR / exp.exp_name
        print_acc_metrics(exp_dir, exp.remove_labels)


