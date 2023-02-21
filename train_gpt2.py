import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

NEED_TOKEN = 'xNeed'
EFFECT_TOKEN = 'xEffect'
INTENT_TOKEN = 'xIntent'
REACT_TOKEN = 'xReact'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_jsonl', default='graph_mrph.jsonl')
    parser.add_argument('--model_name_or_path', default='nlp-waseda/gpt2-small-japanese')
    parser.add_argument('--output_dir', default='comet_gpt2')
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--num_epochs', default=3, type=int)
    args = parser.parse_args()

    dataset = load_dataset('json', data_files=args.graph_jsonl, split='train')
    raw_datasets = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    special_tokens_dict = {
        'additional_special_tokens': [
            NEED_TOKEN,
            EFFECT_TOKEN,
            INTENT_TOKEN,
            REACT_TOKEN,
        ]
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)


    def preprocess_function(examples):
        outputs = []

        for head_text, inf_type_dict in zip(examples['event'], examples['inference']):
            for inf_type, inf_dir_dict in inf_type_dict.items():
                if inf_dir_dict is None:
                    continue

                for inf_dir, tail_list in inf_dir_dict.items():
                    if tail_list is None:
                        continue

                    if inf_type == 'event':
                        if inf_dir == 'before':
                            rel_token = NEED_TOKEN
                        else:
                            rel_token = EFFECT_TOKEN
                    else:
                        if inf_dir == 'before':
                            rel_token = INTENT_TOKEN
                        else:
                            rel_token = REACT_TOKEN

                    for tail_text in tail_list:
                        output = head_text + rel_token + tail_text + tokenizer.eos_token
                        outputs.append(output)

        return {'data': outputs}


    preprocessed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    tokenized_datasets = preprocessed_datasets.map(
        lambda examples: tokenizer(
            examples['data'],
            truncation=True,
            max_length=args.max_length,
        ),
        batched=True,
        remove_columns=preprocessed_datasets['train'].column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        num_train_epochs=args.num_epochs,
        logging_strategy='epoch',
        save_strategy='no',
    )


    class DataCollatorForComet(DataCollatorForLanguageModeling):
        def torch_call(self, examples):
            batch = super().torch_call(examples)            
            labels = batch['labels']

            # consider eos
            eos_mask = labels == -100
            eos_mask[:, 1:] = eos_mask[:, 1:] ^ eos_mask[:, :-1]
            labels[eos_mask] = self.tokenizer.eos_token_id

            # ignore h and r
            rel_mask = labels >= len(tokenizer)
            tail_mask = (rel_mask.cumsum(dim=-1) - rel_mask.to(int)).to(bool)
            labels[~tail_mask] = -100

            batch['labels'] = labels
            return batch


    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForComet(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model()


if __name__ == '__main__':
    main()
