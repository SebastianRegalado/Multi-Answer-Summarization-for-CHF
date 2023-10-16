from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import ZeroShotClassificationExplainer

tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-base")

model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-base")


zero_shot_explainer = ZeroShotClassificationExplainer(model, tokenizer)


word_attributions = zero_shot_explainer(
    "Today apple released the new Macbook showing off a range of new features found in the proprietary silicon chip computer. ",
    labels = ["finance", "technology", "sports"],
)

import ipdb; ipdb.set_trace()