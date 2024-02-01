from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers import TrainingArguments


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import OPTForCausalLM
from datasets import load_dataset

class DataCollator:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances) -> Any:
        input_ids = [instance['text'] for instance in instances]
        input_pt = self.tokenizer(input_ids,return_tensors='pt',padding="longest",max_length=2048,truncation=True)
            
        batch = dict(
            input_ids=input_pt.input_ids,
            attention_mask=input_pt.attention_mask,
            labels = input_pt.input_ids.clone()
        )
        return batch
    

if __name__ == '__main__':
    model = OPTForCausalLM.from_pretrained("/home/user/opt-model/")
    tokenizer = AutoTokenizer.from_pretrained("/home/user/opt-model/")

    model.resize_token_embeddings(len(tokenizer))

    # prompt = "Hey, are you consciours? Can you talk to me?"
    # inputs = tokenizer(prompt, return_tensors="pt")

    # # Generate
    # generate_ids = model.generate(inputs.input_ids, max_length=30)
    # tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    dataset = load_dataset("/home/user/.cache/huggingface/datasets/YeungNLP___firefly-pretrain-dataset/")
    dataset = dataset.shuffle(seed=7)

    training_args = TrainingArguments(
        bf16=True,
        optim='adamw_torch',
        output_dir='output/',
        per_device_train_batch_size=4,
        warmup_ratio = 0.03,
        learning_rate = 2e-5,
        logging_steps = 10,
        report_to = "tensorboard",
        remove_unused_columns=False
    )

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args = training_args,
        train_dataset=dataset['train'],
        data_collator=DataCollator(tokenizer=tokenizer)
    )

    trainer.train()