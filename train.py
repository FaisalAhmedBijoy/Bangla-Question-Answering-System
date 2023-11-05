import json
import random
import glob
import torch
from itertools import chain
from functools import partial
from transformers import TrainingArguments, Trainer, DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
# from torch.optim.lr_scheduler import LinearWarmup

max_length = 512

def read_json_file(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data



bn_train_files = 'data/train_bangla_samples_fixed_preprocessed.json'
bn_val_files = 'data/train_bangla_samples_fixed_preprocessed.json'


bn_train_data = read_json_file(bn_train_files)
bn_val_data = read_json_file(bn_val_files)

print(f"Lenght of Training Data {len(bn_train_data)}")
print(f"Lenght of Validation Data {len(bn_val_data)}")
model_name='deepset/xlm-roberta-large-squad2'
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def data_preprocessing(dataset):
#     dataset = json.load(open(file_path))
    contexts = []
    questions = []
    answers = []
    for group in dataset:
#     for group in dataset['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers

train_contexts, train_questions, train_answers = data_preprocessing(bn_train_data)
test_contexts, test_questions, test_answers = data_preprocessing(bn_val_data)

squad_train = {'answers': train_answers,'context': train_contexts, 'question': train_questions}
squad_test = {'answers': test_answers,'context': test_contexts, 'question': test_questions}

print(f"Lenght of train_answers Data {len(train_answers)}")
print(f"Lenght of test_answers Data {len(test_answers)}")

def preprocess_function(examples, tokenizer):
#     print(examples.keys())
#     exit()
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
#         print(answer)
        start_char = answer["answer_start"]
        end_char = answer["answer_start"] + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_squad_train = preprocess_function(squad_train,tokenizer)
tokenized_squad_test = preprocess_function(squad_test,tokenizer)


def generated_dict(tokenized_squad):
    tokenized_squad_data = list(map(lambda x: {
            "input_ids": x[0], 
            "attention_mask": x[1], 
            "start_positions": x[2], 
            "end_positions": x[3]
        },
        zip(
            tokenized_squad["input_ids"], 
            tokenized_squad["attention_mask"],
            tokenized_squad["start_positions"], 
            tokenized_squad["end_positions"]
        )
    ))
    return tokenized_squad_data

add_part = partial(generated_dict)

tokenized_squad_train_new = add_part(tokenized_squad_train)
tokenized_squad_val_new = add_part(tokenized_squad_test)



data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=3e-5,
    warmup_ratio=0.1,
    per_device_train_batch_size=38,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    eval_steps=1000,
    save_total_limit=1
    # no_deprecation_warning=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_squad_train_new,
    eval_dataset=tokenized_squad_val_new,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# # trainer.train()
trainer.train()
# trainer.train("./results/checkpoint-1000")
print('training complete')

