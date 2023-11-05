# Bangla_Roberta_Question_and_Answer
This Bangla Question Answering model has been using Roberta Model, which is currently trained on a small set of human-annotated data. For training this model, the Bangla QA data has been converted into the SQuAD v2 format as well as preprocessed. The dataset contains 2504 question-answer pairs.


Huggingface Link: https://huggingface.co/saiful9379/Bangla_Roberta_Question_and_Answer


# Requirements,

Please check this ```requirements.txt```

# Dataset

Here the dataset containts 2504 question-answer and Structure like squad v2.

Example:
```
{
    "data": [
            "title": "অ্যাপল ইনকর্পোরেটেড",
            "paragraphs": [
                {
                    "context": " অ্যাপল ইনকর্পোরেটেড এর সদর দপ্তর ক্যালির্ফোনিয়ার মধ্য সিলিকন ভ্যালীতে অবস্থিত। ৮৫০,০০ বর্গ ফিট জায়গাতে ৬টি বিল্ডিংয়ে অ্যাপলের বর্তমান সদর ১৯৯৩ সালে থেকে অবস্থান করছে।",
                    "qas": [
                        {
                            "question": "অ্যাপল ইনকর্পোরেটেডের সদর দপ্তর কোথায় ?",
                            "id": "",
                            "is_impossible": "",
                            "answers": [
                                {
                                    "answer_start": 34,
                                    "text": "ক্যালির্ফোনিয়ার মধ্য সিলিকন ভ্যালীতে"
                                }
                            ]
                        }
                    ]
                }
            ]
    ]
}

```

# Training
Please check this ```Bangla_Roberta_QA.ipynb``` file

__Configuration__
```
max_length=512
evaluation_strategy="steps",
learning_rate=2e-5,
per_device_train_batch_size=44,
per_device_eval_batch_size=16,
num_train_epochs=300,
weight_decay=0.01,
eval_steps=1000,
save_total_limit=1

```
# Evalution

Please check this ```Evaluation.ipynb``` file

# Inference,


```
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline



model = AutoModelForQuestionAnswering.from_pretrained("saiful9379/Bangla_Roberta_Question_and_Answer")
tokenizer = AutoTokenizer.from_pretrained("saiful9379/Bangla_Roberta_Question_and_Answer")

context = "বাংলাদেশ ও ভারতের অনেক বৃহৎ নদী পূর্ব থেকে পশ্চিমে প্রবাহিত হয়ে বঙ্গোপসাগরে পতিত হয়েছে।\
তন্মধ্যে উত্তরদিক থেকে গঙ্গা, মেঘনা এবং ব্রহ্মপুত্র; দক্ষিণদিক থেকে মহানদী, গোদাবরী, কৃষ্ণা, ইরাবতী এবং কাবেরী নদী উল্লেখযোগ্য।\
৬৪ কিলোমিটারব্যাপী (৪০ মাইল) কৌম নদী সবচেয়ে ছোট নদী হিসেবে সরু খাল দিয়ে এবং ২,৯৪৮ কিলোমিটারব্যাপী (১,৮৩২ মাইল)\
বিশ্বের ২৮তম দীর্ঘ নদী হিসেবে ব্রহ্মপুত্র নদ বাংলাদেশ, চীন, নেপাল ও ভারতের মধ্য দিয়ে প্রবাহিত হয়ে বঙ্গোপসাগরে মিলিত হয়েছে।\
সুন্দরবন ম্যানগ্রোভ বনাঞ্চল গঙ্গা, ব্রহ্মপুত্র ও মেঘনা নদীর ব-দ্বীপকে ঘিরে গঠিত হয়েছে। মায়ানমারের (সাবেক বার্মা) ইরাওয়াদি (সংস্কৃত ইরাবতী)\
নদীও এ উপসাগরে মিলিত হয়েছে এবং একসময় গভীর ও ঘন ম্যানগ্রোভ বনাঞ্চলের সৃষ্টি করেছিল।"
 
question = "ব্রহ্মপুত্র নদের মোট দৈর্ঘ্য কত ?"

QA = pipeline('question-answering', model=model, tokenizer=tokenizer)
QA_input = {'question': question,'context':context}

prediction = QA(QA_input)
print(prediction)

```

```
@misc{Bangla_Robert_QA ,
  title={Transformer Based Bangla_Robert_QA},
  author={Md Saiful Islam},
  howpublished={},
  year={2023}
}
```
