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








def data_preprocessing(dataset):
#     dataset = json.load(open(file_path))
    contexts = []
    questions = []
    answers = []
   
    
    # for group in dataset:
    for group in dataset['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers

def preprocess_function(examples, tokenizer):
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



if __name__=='__main__':
    
    train_json_path = 'data/valid_bangla_samples_fixed_preprocessed.json'
    val_json_path = 'data/valid_bangla_samples_fixed_preprocessed.json'
    model_name='saiful9379/Bangla_Roberta_Question_and_Answer'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    train_data = read_json_file(train_json_path)

    
    val_data = read_json_file(val_json_path)

    print(f"Lenght of Training Data {len(train_data)}")
    print(f"Lenght of Validation Data {len(val_data)}")
    train_contexts, train_questions, train_answers = data_preprocessing(train_data)
    test_contexts, test_questions, test_answers = data_preprocessing(val_data)

    squad_train = {'answers': train_answers,'context': train_contexts, 'question': train_questions}
    squad_test = {'answers': test_answers,'context': test_contexts, 'question': test_questions}

    print(f"Lenght of train_answers Data {len(train_answers)}")
    print(f"Lenght of test_answers Data {len(test_answers)}")
    add_part = partial(generated_dict)
    tokenized_squad_train = preprocess_function(squad_train,tokenizer)
    tokenized_squad_test = preprocess_function(squad_test,tokenizer)


    tokenized_squad_train_new = add_part(tokenized_squad_train)
    tokenized_squad_val_new = add_part(tokenized_squad_test)
    print('tokenized squad val new :',len(tokenized_squad_val_new))





