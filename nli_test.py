from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = 'roberta-large-mnli'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = "This sounds like cancer."
hypothesis = "This doesn't sound like cancer."

# encode sequences
tokenized = tokenizer(text, hypothesis, return_tensors="pt")

# perform inference
logits = model(**tokenized)[0]
probs = logits.softmax(dim=1)
print(probs)
