import pandas as pd
from openai_test import ConversationGenerator, CONFIG_DIR, OUTPUT_DIR
from openai import OpenAI
from utils import DATA_DIR
from sklearn.model_selection import train_test_split

SUFFIX = "11_10_basic"
CONFIG_PATH = CONFIG_DIR / f"zero_shot_context_basic.yaml"

def create_finetuning_data(input_path: str, config_path: str):
    generator = ConversationGenerator(config_path)
    data = pd.read_csv(input_path)
    data = data[data["s_perspective_1"].notnull()]
    valid_perspectives = set(generator.perspective_map.keys())
    data = data[data["s_perspective_1"].isin(valid_perspectives)]

    data_split = train_test_split(data, test_size=0.2, random_state=42)
    prefixes = ["train", "test"]
    
    output_paths = []
    for data, prefix in zip(data_split, prefixes):
        data.to_csv(DATA_DIR / f"openai_{prefix}_data_{SUFFIX}.csv", index=False)
        output_path = OUTPUT_DIR / f"{prefix}_{generator.name}_{SUFFIX}.jsonl"
        finetuning_data = []
        for _, row in data.iterrows():
            question = row["subject"]
            context = row["content"]
            answer = row["sentence"]
            persepctive = row["s_perspective_1"]
            conversation = generator.create_conversation_with_perspective(question, context, answer, persepctive)
            finetuning_data.append({"messages": conversation})

        ### save finetuning data as jsonl file
        import jsonlines
        with jsonlines.open(output_path, mode="w") as writer:
            writer.write_all(finetuning_data)
        output_paths.append(output_path)

    return output_paths
    

if __name__ == "__main__":
    input_path = DATA_DIR / f"yahoo_health_labeled_{SUFFIX}.csv"
    output_paths = create_finetuning_data(input_path, CONFIG_PATH)
    client = OpenAI()

    ids = []
    for output_path in output_paths:
        id = client.files.create(
            file=open(output_path, "rb"),
            purpose="fine-tune"
        ).id
        ids.append(id)

    for id in ids:
        file = client.files.retrieve(id)
        print(file.id, file.filename)