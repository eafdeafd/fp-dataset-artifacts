import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import torch
import torch.nn as nn

import wandb
#wandb.init(mode="disabled")           
NUM_PREPROCESSING_WORKERS = 2

# python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/ --save_steps 1000 --save_total_limit 1 --load_best_model_at_end --logging_steps 1000 --per_device_train_batch_size 64 --evaluation_strategy steps
# python3 run.py --do_train --task nli --dataset ./data/contrast+orig/train.tsv --output_dir ./trained_model/ --save_steps 500 --save_total_limit 1 --load_best_model_at_end --logging_steps 500 --per_device_train_batch_size 256 --evaluation_strategy steps --per_device_eval_batch_size 256
# python3 run.py --do_eval --task nli --dataset data/contrast+orig/test.tsv --model ./trained_model/checkpoint-24000/ --output_dir ./eval_model/
# python3 run.py --do_train --task nli --dataset contrast-set --output_dir ./trained_model/debiased_snli --save_steps 500 --save_total_limit 1 --load_best_model_at_end --logging_steps 500 --per_device_train_batch_size 256 --evaluation_strategy steps --per_device_eval_batch_size 256  --original True --debias True --biased_model ./trained_model/hypothesis_only_snli/
# python3 run.py --do_eval --task nli --dataset contrast-set --output_dir ./eval_model/debiased_snli --original True --model ./trained_model/debiased_snli/checkpoint-5500
def main():

    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    argp.add_argument('--hard', type=bool, default=False)
    argp.add_argument('--hypothesis_only', type=bool, default=False)
    argp.add_argument('--original', type=bool, default=False)
    argp.add_argument('--debias', type=bool, default=False)
    argp.add_argument('--biased_model', type=str, default='./trained_model/hypothesis_only_snli')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    elif args.dataset == ('contrast-set'):
        # load it, preprocess, assumes no missing labels
        def clean_data(dataset):
            if args.hard:
                dataset = dataset.filter(lambda example: example["captionID"] != "original")
            if args.original:
                dataset = dataset.filter(lambda example: example["captionID"] == "original")
            dataset = dataset.filter(lambda example: example["sentence1"] is not None and len(example["sentence1"]) > 0 and example["sentence2"] is not None and len(example["sentence2"]) > 0 and example["gold_label"] is not None)
            dataset = dataset.select_columns(['index', 'captionID', 'sentence1', 'sentence2', 'gold_label'])
            dataset = dataset.rename_column('sentence1',  'premise')
            dataset = dataset.rename_column('sentence2', 'hypothesis')
            dataset = dataset.rename_column('gold_label', 'label')
            label_map = {"contradiction": 2,
                        "neutral": 1,
                        "entailment": 0}
            if args.hypothesis_only:
                dataset = dataset.map(lambda example: {"premise":""})
            dataset = dataset.map(lambda example: {"label": label_map[example["label"]]})
            dataset = dataset.cast_column("premise", datasets.Value(dtype='string', id='None'))
            dataset = dataset.cast_column("hypothesis", datasets.Value(dtype='string', id='None'))
            dataset = dataset.cast_column("label", datasets.Value(dtype='int64', id='None'))
            return dataset
        dataset_id = None
        if not training_args.do_train and training_args.do_eval:
            eval_split = 'test'
            data_files = {
                "test": "./data/contrast+orig/test.tsv"
            }
        else:
            eval_split = 'validation'
            data_files = {
                "train": "./data/contrast+orig/train.tsv",
                "validation": "./data/contrast+orig/dev.tsv",
                "test": "./data/contrast+orig/test.tsv"
            }
        dataset = datasets.load_dataset('csv', data_files=data_files, delimiter='\t')
        for d in data_files:
            dataset[d] = clean_data(dataset[d])
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)
    
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head

    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    if args.debias:
        biased_model = model_class.from_pretrained(args.biased_model, **task_kwargs)
    model = model_class.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )
    
    def preprocess_for_biased_model(input_ids, device):        
        # Identify positions of the first and second [SEP] tokens
        sep_positions = (input_ids == 102).to(device).long().cumsum(dim=1)
        # We are interested in keeping everything after the first [SEP] token
        # Hence, we create a mask that zeroes out everything up to and including the first [SEP] token
        keep_mask = sep_positions >= 1
        processed_input_ids = input_ids * keep_mask
        processed_input_ids[:, 0] = 101
        processed_input_ids = processed_input_ids.gather(1, (processed_input_ids == 0.0).to(device).long().sort(dim=1, stable=True)[1])
        processed_input_ids.to(device)
        return processed_input_ids
    
    class ResidualTrainer(Trainer):
        def __init__(self, bias_model,*args, **kwargs):
            super().__init__(*args, **kwargs)
            self.biased_model = bias_model


        def compute_loss(self, model, inputs, return_outputs=False):    
            input_ids = inputs["input_ids"]
            self.biased_model.to(model.device)
            biased_inputs = inputs.copy()
            biased_inputs["input_ids"] = preprocess_for_biased_model(biased_inputs["input_ids"], model.device)
            with torch.no_grad():
                biased_logits = self.biased_model(**biased_inputs).get("logits")
            out = model(**inputs)
            combined_logits = out.get("logits") + biased_logits
            labels = inputs['labels']
            lf = nn.CrossEntropyLoss()
            loss = lf(combined_logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, out) if return_outputs else loss


    # Select the training configuration
    trainer_class = Trainer if not args.debias else ResidualTrainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = datasets.load_metric('squad')
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        compute_metrics = compute_accuracy
    

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)
    

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    if not args.debias:
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset_featurized,
            eval_dataset=eval_dataset_featurized,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_and_store_predictions,
        )
    else:
        trainer = trainer_class(
            bias_model=biased_model,
            model=model,
            args=training_args,
            train_dataset=train_dataset_featurized,
            eval_dataset=eval_dataset_featurized,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_and_store_predictions,
        )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    main()
