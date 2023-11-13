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
from utils import DATA_DIR, EXP_DIR

TEXT_COL = "sentence"
CONTEXT_COL = "context"
LABEL_COL = "s_perspective_1"
LABEL_MAP = {
    'suggestion': 0,
    'cause': 1,
    'information': 2,  
    'experience': 3,
    'treatment': 4,
    'odd': 5,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

TEMPLATE_MAP = {
    "full": Template("Subject: $subject\nContent: $content\nAnswer: $answer"),
    "context": Template("$subject\n$content"),
    "subject_only": Template("$subject"),
    "none": Template(""),
}
TEMPLATE_NAMES = list(TEMPLATE_MAP.keys())

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 4
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

class ExperimentConfig:
    def __init__(self, template_name: str, remove_labels: T.List[str] = [], freeze_model: bool = False):
        self.template_name = template_name
        self.template = TEMPLATE_MAP[template_name]
        self.remove_labels = remove_labels
        self.freeze_model = freeze_model
        self.exp_name = self.create_exp_name()
        self.exp_dir = EXP_DIR / self.exp_name

    def create_exp_name(self):
        exp_name = "finetune_"
        exp_name += "wo_" if self.remove_labels else ""
        exp_name += "_".join(self.remove_labels)
        exp_name += "_" if self.remove_labels else ""
        exp_name += self.template_name
        exp_name += "_freeze" if self.freeze_model else ""
        return exp_name

def create_context(data: pd.DataFrame, template: Template):
    subject = data["subject"]
    content = str(data["content"])
    answer = data["answer"]
    context = template.safe_substitute(subject=subject, content=content, answer=answer)
    return context

def tokenize_generator(tokenizer: AutoTokenizer):
    def tokenize_batch(batch):
        return tokenizer(batch[TEXT_COL], batch[CONTEXT_COL], padding="max_length", truncation=True, return_tensors="pt")
    return tokenize_batch

def load_dataset(data: pd.DataFrame, tokenizer: AutoTokenizer) -> Dataset:
    tokenize_batch = tokenize_generator(tokenizer)
    dataset = Dataset.from_pandas(data, preserve_index=False)
    dataset = dataset.map(tokenize_batch, batched=True)
    return dataset

def load_data(file_path: Path, template: Template, remove_labels: T.List[str] = []):
    data = pd.read_csv(file_path)
    data[CONTEXT_COL] = data.apply(lambda row: create_context(row, template), axis=1)
    data = data[[TEXT_COL, CONTEXT_COL, LABEL_COL]]
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
    CLASS_WEIGHTS=torch.tensor(class_weights,dtype=torch.float).to("cuda")

    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return Data(train_data, test_data)

def finetune(
    exp: ExperimentConfig, data: Data, remove_labels: T.List[str] = [], num_epochs: int = 20
):

    exp_dir = exp.exp_dir
    # Initialize the tokenizer and model
    num_labels = len(LABEL_MAP) - len(remove_labels)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # Freeze the model's base layers
    if exp.freeze_model:
        for name, param in model.named_parameters():
            prefixes = ["bert.embeddings", "bert.encoder", "transformer"]
            if any(name.startswith(p) for p in prefixes):
                param.requires_grad = False

    # # Print the names of the model's parameters
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
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=2e-3 if exp.freeze_model else 2e-5,
        weight_decay=0.01,
        logging_dir= exp_dir / "logs",
        logging_steps=50,
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
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

def eval_test_data(exp_dir: Path, template: Template, test_file: Path) -> pd.DataFrame:
    # Load the trained model
    model_dir = exp_dir / "model"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_data = pd.read_csv(test_file)
    test_data[CONTEXT_COL] = test_data.apply(lambda row: create_context(row, template), axis=1)
    test_data = test_data[[TEXT_COL, CONTEXT_COL, LABEL_COL]]
    test_data = test_data[test_data[LABEL_COL].notna()]
    test_data = test_data.rename(columns={LABEL_COL: "label"})

    ### one hot encoding of labels
    labels = test_data["label"].unique()
    labels = {label: i for i, label in enumerate(labels)}
    test_data["label"] = test_data["label"].map(LABEL_MAP)

    test_dataset = load_dataset(test_data, tokenizer)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(exp_dir),
        train_dataset=test_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda p: {
            "accuracy": (p.predictions.argmax(1) == p.label_ids).mean(),
        },
    )
    results = trainer.predict(test_dataset)
    predictions = results.predictions.argmax(-1)
    test_data["predictions"] = predictions
    test_data["predictions"] = test_data["predictions"].map(INV_LABEL_MAP)
    test_data["label"] = test_data["label"].map(INV_LABEL_MAP)

    # Print accuracy as percentage
    labels = list(test_data["label"].unique())
    acc = (test_data["label"] == test_data["predictions"]).mean()
    print(f"Accuracy: {acc * 100:.2f}%")
    print(labels)
    print(confusion_matrix(test_data["label"], test_data["predictions"], labels=labels))
    print(f1_score(test_data["label"], test_data["predictions"], labels=labels, average=None))
    weighted_f1 = f1_score(test_data["label"], test_data["predictions"], labels=labels, average="weighted")
    print(f"Weighted F1: {weighted_f1:.3f}")

def print_acc_metrics(exp_dir: Path, remove_labels: T.List[str] = []):
    labels = [l for l in list(LABEL_MAP.keys()) if l not in remove_labels]
    prediction_data = pd.read_csv(exp_dir / "test_dataset_labeled.csv")
    acc = (prediction_data["label"] == prediction_data["predictions"]).mean()

    # Print accuracy as percentage
    print(f"Accuracy: {acc * 100:.2f}%")
    print(labels)
    print(confusion_matrix(prediction_data["label"], prediction_data["predictions"], labels=labels))
    print(f1_score(prediction_data["label"], prediction_data["predictions"], labels=labels, average=None))
    weighted_f1 = f1_score(prediction_data["label"], prediction_data["predictions"], labels=labels, average="weighted")
    print(f"Weighted F1: {weighted_f1:.3f}")

def run_experiments(experiments: T.List[ExperimentConfig], num_epochs: int = 20):
    for exp in experiments:
        print(exp.exp_name)
        data_file = DATA_DIR / "yahoo_health_labeled_11_10_basic.csv"
        data = load_data(data_file, exp.template, exp.remove_labels)
        finetune(exp, data, exp.remove_labels, num_epochs)
        eval_data(exp.exp_dir, data)

if __name__ == "__main__":
    experiments = [
        ExperimentConfig(template_name, remove_labels, freeze_model)
        for remove_labels in [["odd"]]
        for template_name in ["context", "subject_only"]
        for freeze_model in [False]
    ]
    # run_experiments(experiments, num_epochs=5)
    for exp in experiments:
        print(exp.exp_name)
        eval_test_data(exp.exp_dir, exp.template, DATA_DIR / "openai_test_data_11_10_basic.csv")
        # print_acc_metrics(exp.exp_dir, exp.remove_labels)


