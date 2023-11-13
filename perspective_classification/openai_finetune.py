import os
import openai
from openai import OpenAI

TRAIN_FILE = "file-XLi0iVfg4iyl7DpPooNZXSML"
VAL_FILE = "file-VRXlEe5N57tYHrzIkPz7AyL4"

API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY

def finetune(model: str = "gpt-3.5-turbo"):
    client = OpenAI()
    job = client.fine_tuning.jobs.create(
        training_file=TRAIN_FILE,
        validation_file=VAL_FILE,
        hyperparameters={
            "batch_size": 8,
            "n_epochs": 2,

        },
        model=model,
    )
    print(job)

if __name__ == "__main__":
    finetune() 
