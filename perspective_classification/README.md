## Finetuning

The file `bert_finetune.py` shows how to finetune a BERT model on the perspective classification task. The file `openai_finetune.py` shows how to finetune a GPT3.5 Turbo model on the same task. The file `openai_test.py` shows how to test a GPT3.5 Turbo model on the same task. 

The other files in this folder are baselines or models that require no finetuning.

## Classification

Since we chose GPT3.5 Turbo as our best model, you can use the classify script to classify the perspectives of your choice. Simply change the file to classify in the `classify.py` file and run the script. Remember to change the model id in `openai_test.py` to the id of your model.