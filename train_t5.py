import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

NEED_PREFIX = '次の出来事に必要な前提条件は何ですか: '
EFFECT_PREFIX = '次の出来事の後に起こりうることは何ですか: '
INTENT_PREFIX = '次の出来事が起こった動機は何ですか: '
REACT_PREFIX = '次の出来事の後に感じることは何ですか: '


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_jsonl', default='graph.jsonl')
    parser.add_argument('--model_name_or_path', default='megagonlabs/t5-base-japanese-web')
    parser.add_argument('--output_dir', default='comet_t5')
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--num_epochs', default=3, type=int)
    args = parser.parse_args()

    dataset = load_dataset('json', data_files=args.graph_jsonl, split='train')
    raw_datasets = dataset.train_test_split(test_size=0.1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    max_input_length = args.max_length
    max_target_length = args.max_length


    def preprocess_function(examples):
        inputs = []
        targets = []

        for head_text, inf_type_dict in zip(examples['event'], examples['inference']):
            for inf_type, inf_dirs in inf_type_dict.items():
                if inf_dirs is None:
                    continue

                for inf_dir, tail_texts in inf_dirs.items():
                    if tail_texts is None:
                        continue

                    if inf_type == 'event':
                        if inf_dir == 'before':
                            prefix = NEED_PREFIX
                        else:
                            prefix = EFFECT_PREFIX
                    else:
                        if inf_dir == 'before':
                            prefix = INTENT_PREFIX
                        else:
                            prefix = REACT_PREFIX

                    for tail_text in tail_texts:
                        inputs.append(prefix + head_text)
                        targets.append(tail_text)

        model_inputs = tokenizer(
            inputs,
            truncation=True,
            max_length=max_input_length,
        )

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                truncation=True,
            )

        model_inputs['labels'] = labels['input_ids']
        return model_inputs


    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    args = Seq2SeqTrainingArguments(
        args.output_dir,
        evaluation_strategy='epoch',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        num_train_epochs=args.num_epochs,
        logging_strategy='epoch',
        save_strategy='no',
        # fp16=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
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
