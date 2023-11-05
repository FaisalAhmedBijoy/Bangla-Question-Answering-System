from itertools import chain
from functools import partial
from transformers import TrainingArguments, Trainer, DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
# from torch.optim.lr_scheduler import LinearWarmup
from data_processing import read_json_file,data_preprocessing,preprocess_function,generated_dict



def data_processing_from_json_to_tokenization(train_json_path,val_json_path,model,tokenizer):
    train_data = read_json_file(train_json_path)
    val_data = read_json_file(val_json_path)
    train_contexts, train_questions, train_answers = data_preprocessing(train_data)
    test_contexts, test_questions, test_answers = data_preprocessing(val_data)

    squad_train = {'answers': train_answers,'context': train_contexts, 'question': train_questions}
    squad_test = {'answers': test_answers,'context': test_contexts, 'question': test_questions}


    add_part = partial(generated_dict)
    tokenized_squad_train = preprocess_function(squad_train,tokenizer)
    tokenized_squad_test = preprocess_function(squad_test,tokenizer)


    tokenized_squad_train_new = add_part(tokenized_squad_train)
    tokenized_squad_val_new = add_part(tokenized_squad_test)
    # print('tokenized squad val new :',len(tokenized_squad_val_new))
    return tokenized_squad_train_new,tokenized_squad_val_new

def model_training(tokenized_squad_train_new,tokenized_squad_val_new,model,tokenizer,model_saved_path,training_logs):
    
    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir=training_logs,
        evaluation_strategy="steps",
        learning_rate=3e-5,
        warmup_ratio=0.1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        eval_steps=10
        # save_total_limit=1
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
    trainer.train()
    trainer.save_model(model_saved_path)
    print('training complete')
    return trainer
    
    # trainer.train("./results/checkpoint-1000")

if __name__ == '__main__':
    train_json_path = 'data/valid_bangla_samples_fixed_preprocessed.json'
    val_json_path = 'data/valid_bangla_samples_fixed_preprocessed.json'
    model_name='saiful9379/Bangla_Roberta_Question_and_Answer'
    model_saved_path="logs/bangla_roberta_model"
    training_logs='logs/training_logs'
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenized_squad_train_new,tokenized_squad_val_new=data_processing_from_json_to_tokenization(train_json_path,val_json_path,model,tokenizer)
    trainer=model_training(tokenized_squad_train_new,tokenized_squad_val_new,model,tokenizer,model_saved_path,training_logs)
    


