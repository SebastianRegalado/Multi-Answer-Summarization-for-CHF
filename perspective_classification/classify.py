import json
from pathlib import Path
from utils import DATA_DIR
from tqdm import tqdm
from perspective_classification.openai_test import ConversationGenerator, CONFIG_DIR, BEST_MODEL


def classify_sentences(input_file: Path, config_file: Path, output_file: Path):

    with open(input_file, 'r') as f:
        data = json.load(f)
    
    generator = ConversationGenerator(config_file, BEST_MODEL)

    save_every = 50
    count = 0
    for question in tqdm(data):
        subject = question['subject']
        content = question['content']

        classified_answers = []
        for answer in question['answers']:
            classified_answer = []
            for sentence in answer:
                conversation = generator.create_conversation(subject, content, sentence)
                response = generator.chat_with_gpt(conversation)
                classified_answer.append((sentence, response))
            classified_answers.append(classified_answer)
        question['classified_answers'] = classified_answers

        count += 1
        if count % save_every == 0:
            with open(output_file, 'w') as f:
                json.dump(data, f)


if __name__ == '__main__':
    input_file = DATA_DIR / "yahoo_health_medical_short.json"
    config_file = CONFIG_DIR / "zero_shot_context.yaml"
    output_file = DATA_DIR / "yahoo_health_medical_short_classified.json"
    classify_sentences(input_file, config_file, output_file)