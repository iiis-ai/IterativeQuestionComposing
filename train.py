import logging
import os
import sys
import itertools
from typing import Any, Union, Tuple

import torch
import datasets
import transformers
from transformers import (
    AutoConfig,
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    set_seed,
    LlamaConfig,
    BatchEncoding,
)
from datasets import (
    load_dataset,
    load_from_disk,
    DatasetDict,
    Dataset,
    IterableDatasetDict,
    IterableDataset,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential


from utils.arguments import (
    ModelArguments,
    DataTrainingArguments,
    TrainingArguments,
    print_args,
)
from utils.misc import log_func_time
from utils.prompter import BasePrompter, PROMPTER_DICT

logger = logging.getLogger(__name__)

UnionDatasetType = Union[
    DatasetDict, Dataset, IterableDatasetDict, IterableDataset
]
ArgTuple = Tuple[
    ModelArguments,
    DataTrainingArguments,
    TrainingArguments,
]


def parse_args() -> ArgTuple:
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
        )
    )
    if sys.argv[-1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        args = parser.parse_args_into_dataclasses()
    (model_args, data_args, training_args) = args
    return model_args, data_args, training_args


class LlamaTrainer:
    def __init__(
            self,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            training_args: TrainingArguments,
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        self._model_config: LlamaConfig | None = None

        self._prompter: BasePrompter | None = None
        self._tokenizer = None
        self._data_collator: Any | None = None

        self.train_dataset: datasets.Dataset | None = None
        self.eval_dataset: datasets.Dataset | None = None

        self.model = None
        self.trainer: Trainer | None = None

    def initialize(self):
        self.seed_everything()

        self.load_model_config()

        self.load_tokenizer()
        self.load_prompter()
        self.load_data_collator()

        self.process_dataset()
        self.model = self.load_base_model()
        self.load_trainer()
        self.set_wandb()

    @property
    def model_config(self) -> LlamaConfig:
        if self._model_config is None:
            self._model_config = self.load_model_config()
        return self._model_config

    @property
    def prompter(self) -> BasePrompter:
        if self._prompter is None:
            self._prompter = self.load_prompter()
        return self._prompter

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.load_tokenizer()
        return self._tokenizer

    @property
    def data_collator(self) -> Any:
        if self._data_collator is None:
            self._data_collator = self.load_data_collator()
        return self._data_collator

    def seed_everything(self) -> None:
        set_seed(self.training_args.seed)

    @log_func_time
    def load_model_config(self) -> LlamaConfig:
        model_args = self.model_args

        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            **config_kwargs
        )
        return model_config

    @log_func_time
    def load_tokenizer(self):
        model_args = self.model_args
        if model_args.tokenizer_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                trust_remote_code=True,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                trust_remote_code=True,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        if 'Qwen' in model_args.tokenizer_name_or_path:
            tokenizer.pad_token_id = (151646)   # <|extra0|>
        else:
            tokenizer.pad_token_id = (0) # unk. we want this to be different from the eos token
        tokenizer.padding_side = "left"  # Allow batched inference

        if not tokenizer.eos_token_id:
            try:
                tokenizer.eos_token_id = tokenizer.eod_id
                print('Now setting eos_token_id to eod_id for Qwen models')
            except Exception as e:
                raise(f'No "eos_token_id" or "eod_id" for the tokenizer. Please specify one.')

        return tokenizer

    @log_func_time
    def load_prompter(self) -> BasePrompter:
        PrompterClass = PROMPTER_DICT[self.model_args.prompt_templater]
        prompter = PrompterClass(
            self.model_args.prompt_template_name,
            self.model_args.prompt_template_verbose,
        )
        return prompter

    @log_func_time
    def load_data_collator(self) -> Any:
        return transformers.DataCollatorForSeq2Seq(
            self.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        )

    @log_func_time
    def load_base_model(self):
        model_args = self.model_args
        # training_args = self.training_args

        # device_map = "auto"
        # if self.use_ddp:
        #     device_map = {"": training_args.local_rank}
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        
        use_flash_attn_2 = model_args.flash_attn
        if 'Qwen' in model_args.model_name_or_path:
            use_flash_attn_2 = False
            print('For Qwen models, use_flash_attention_2 should not be set to True.')
        @retry(wait=wait_random_exponential(min=1, max=100), stop=stop_after_attempt(10))
        def create_base_model():
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                torch_dtype=torch_dtype,
                use_flash_attention_2=use_flash_attn_2,
                trust_remote_code=True
            )
            return model
        model = create_base_model()
        # use_flash_attention_2=True
        return model

    @property
    def use_ddp(self) -> bool:
        return self.training_args.world_size != 1

    @log_func_time
    def load_dataset(self) -> UnionDatasetType:
        if self.data_args.load_from_disk:
            data = load_from_disk(self.data_args.dataset_dir)
        else:
            data = load_dataset(
                self.data_args.dataset_dir, self.data_args.dataset_config_name,
                cache_dir=self.data_args.data_cache_dir
            )
        return data

    def tokenize(self, text: str, add_eos_token: bool = True) -> BatchEncoding:
        data_args = self.data_args

        result = self.tokenizer(
            text,
            truncation=True,
            max_length=data_args.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < data_args.max_seq_length
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def tokenize_ignore(self, full_prompt: str, prompt_wo_response: str, add_eos_token: bool = True) -> BatchEncoding:
        data_args = self.data_args
        ids_wo_response = self.tokenizer.encode(prompt_wo_response)
        instruct_len = min(len(ids_wo_response), data_args.max_seq_length)
        result = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=data_args.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        if result["input_ids"][-1] != self.tokenizer.eos_token_id and len(result["input_ids"]) < data_args.max_seq_length and add_eos_token:
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        # ignore index is -100
        result["labels"][:instruct_len] = [-100]*instruct_len
        return result

    def tokenize_mix(self, full_prompt: str, prompt_wo_response: str, response: str, instruction: str, add_eos_token: bool = True) -> BatchEncoding:
        data_args = self.data_args
        if instruction == "":
            result = self.tokenizer(response, truncation=True, max_length=data_args.max_seq_length, padding=False, return_tensors=None)
            result['input_ids'] = result['input_ids'][1:]
            result['attention_mask'] = result['attention_mask'][1:]
            if result["input_ids"][-1] != self.tokenizer.eos_token_id and len(result["input_ids"]) < data_args.max_seq_length and add_eos_token:
                result["input_ids"].append(self.tokenizer.eos_token_id)
                result["attention_mask"].append(1)
            result["labels"] = result["input_ids"].copy()          
        else:
            result = self.tokenizer(full_prompt, truncation=True, max_length=data_args.max_seq_length, padding=False, return_tensors=None)
            if result["input_ids"][-1] != self.tokenizer.eos_token_id and len(result["input_ids"]) < data_args.max_seq_length and add_eos_token:
                result["input_ids"].append(self.tokenizer.eos_token_id)
                result["attention_mask"].append(1)
            result["labels"] = result["input_ids"].copy()
            # ignore index is -100
            ids_wo_response = self.tokenizer.encode(prompt_wo_response)
            instruct_len = min(len(ids_wo_response), data_args.max_seq_length)
            result["labels"][:instruct_len] = [-100]*instruct_len
        result["length"] = len(result["input_ids"])
        
        return result


    def tokenize_for_concat(self, text: str, add_eos_token: bool = True) -> BatchEncoding:
        result = self.tokenizer(text)
        if result["input_ids"][-1] != self.tokenizer.eos_token_id and add_eos_token:
            result["input_ids"].append(self.tokenizer.eos_token_id)
            # result["attention_mask"].append(1)
        # result["labels"] = result["input_ids"].copy()
        return result

    def group_texts(self, examples):
        # Concatenate all texts.
        block_size = self.data_args.max_seq_length  #TODO: change to data_args.block_size
        # concatenated_examples = {list(itertools.chain(*examples['input_ids']))}
        whole_id = list(itertools.chain.from_iterable(examples['input_ids']))
        # do not drop. Split to chunks of max_len.
        input_ids = [whole_id[i : i + block_size] for i in range(0, len(whole_id), block_size)]
        at_mask = [[1]*len(input_id) for input_id in input_ids]
        labels = input_ids.copy() # already done in tokenize_for_concat
        return {"input_ids": input_ids, "attention_mask": at_mask, "labels": labels}

    def map_dataset(self, dt: Dataset) -> Dataset:
        data_args = self.data_args
        train_mode = data_args.train_mode
        # old version support
        if data_args.ignore_instruction:
            train_mode == 'ignore'
        elif data_args.concat_all:
            train_mode == 'concat'

        if train_mode == 'mix':
            print("For null instruction, only encode output and remove BOS")
            dt = dt.shuffle().map(self.generate_and_tokenize_prompt_mix, num_proc=self.data_args.num_proc, remove_columns=dt.column_names) 
        elif train_mode == 'concat':
            # dt.column_names removed then group_texts can differ from batch_size
            # transformers/examples/pytorch/language-modeling/run_clm.py used this technique
            print('Concating all data in 1 dim then divide into blocks')
            dt_input_id = dt.shuffle().map(self.generate_and_tokenize_prompt_concat, num_proc=self.data_args.num_proc, remove_columns=dt.column_names)
            dt = dt_input_id.map(self.group_texts, num_proc=self.data_args.num_proc, batched=True, batch_size=2000)
        elif train_mode == 'ignore':
            print('Ignoring instruction in training, standard SFT')
            dt = dt.shuffle().map(self.generate_and_tokenize_prompt_ignore, num_proc=self.data_args.num_proc)
        elif train_mode == 'wo_ignore':
            print('Instruction takes part in training as well, non-standard SFT')
            dt = dt.shuffle().map(self.generate_and_tokenize_prompt, num_proc=self.data_args.num_proc)
        else:
            raise ValueError(f'You must specify a train mode in concat (pretrain), ignore (SFT), wo_ignore')
        if train_mode == 'mix':
            dt = dt.select_columns(['input_ids', 'attention_mask', 'labels', 'length'])
        else:
            dt.select_columns(['input_ids', 'attention_mask', 'labels'])
        return dt

    def generate_and_tokenize_prompt_mix(self, data_point: dict) -> BatchEncoding:
        full_prompt = self.prompter.generate_prompt(data_point)
        instruction = self.prompter.generate_instruction(data_point)
        response = self.prompter.generate_response(data_point)
        prompt_wo_response = self.prompter.generate_wo_response_prompt(data_point)
        tokenized_full_prompt = self.tokenize_mix(full_prompt, prompt_wo_response, response, instruction)
        return tokenized_full_prompt

    def generate_and_tokenize_prompt_ignore(self, data_point: dict) -> BatchEncoding:
        full_prompt = self.prompter.generate_prompt(data_point)
        prompt_wo_response = self.prompter.generate_wo_response_prompt(data_point)
        tokenized_full_prompt = self.tokenize_ignore(full_prompt, prompt_wo_response)
        return tokenized_full_prompt

    def generate_and_tokenize_prompt(self, data_point: dict) -> BatchEncoding:
        full_prompt = self.prompter.generate_prompt(data_point)
        tokenized_full_prompt = self.tokenize(full_prompt)
        return tokenized_full_prompt

    def generate_and_tokenize_prompt_concat(self, data_point: dict) -> BatchEncoding:
        full_prompt = self.prompter.generate_prompt(data_point)
        # tokenizer(text) will output {'input_ids': [], 'attention_mask':[]}
        tokenized_full_prompt = self.tokenize_for_concat(full_prompt)
        return tokenized_full_prompt

    def generate_and_tokenize_prompt_overflow(self, data_point: dict) -> list:
        eos_id = self.tokenizer.eos_token_id
        max_seq_len = self.data_args.max_seq_length
        full_prompt = self.prompter.generate_prompt(data_point)
        whole_id = self.tokenizer.encode(full_prompt)
        id_chunk = [whole_id[i:i + max_seq_len] for i in range(0, len(whole_id), max_seq_len)]
        if len(id_chunk[-1]) < max_seq_len and id_chunk[-1][-1] != eos_id:
            id_chunk[-1].append(eos_id) 
        return {"id_chunk": id_chunk}


    @log_func_time
    def process_dataset(self):
        data_args = self.data_args
        dataset = self.load_dataset()

        if data_args.max_eval_samples > 0:
            if "test" not in dataset.keys():
                dataset = dataset["train"].train_test_split(
                    test_size=data_args.max_eval_samples,
                    shuffle=True,
                    seed=self.training_args.seed,
                )
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
        else:
            train_dataset = dataset["train"]
            eval_dataset = None

        if data_args.max_train_samples > 0:
            train_dataset = train_dataset.shuffle().select(
                range(min(len(train_dataset), data_args.max_train_samples))
            )
        if data_args.max_eval_samples > 0:
            eval_dataset = eval_dataset.shuffle().select(
                range(min(len(eval_dataset), data_args.max_eval_samples))
            )

        self.train_dataset = self.map_dataset(train_dataset)
        print(f'Actual #chunks of train dataset: {len(self.train_dataset)}')
        if eval_dataset:
            self.eval_dataset = self.map_dataset(eval_dataset)
            print(f'Actual #chunks of test dataset: {len(self.eval_dataset)}')
    @log_func_time
    def load_trainer(self):
        assert (
                self.model is not None
        ), "Must load model before loading trainer"
        assert (
                self.train_dataset is not None
        ), "Must load dataset before loading trainer"

        self.trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=self.training_args,
            data_collator=self.data_collator,
        )

    def set_wandb(self):
        if self.data_args.wandb_project:
            os.environ["WANDB_PROJECT"] = self.data_args.wandb_project
        if self.data_args.wandb_name:
            os.environ["WANDB_NAME"] = self.data_args.wandb_name
    @log_func_time
    def train(self):
        self.trainer.model.save_pretrained(self.training_args.output_dir)
        self.tokenizer.save_pretrained(self.training_args.output_dir)
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model(self.training_args.output_dir)
        
        # self.trainer.save_state()

    @property
    def is_logging(self):
        return bool(self.training_args.local_rank == 0)


def test_main():
    model_args = ModelArguments(prompt_template_name="prm800k")
    data_args = DataTrainingArguments(
        dataset_dir="Birchlabs/openai-prm800k-solutions-only",
    )
    training_args = TrainingArguments(output_dir="./llama_marcel")

    lt = LlamaTrainer(model_args, data_args, training_args)
    # lt.initialize()
    return lt


def main():
    model_args, data_args, training_args = parse_args()

    lt = LlamaTrainer(model_args, data_args, training_args)
    if lt.is_logging:
        print_args(model_args, data_args, training_args)

    lt.initialize()
    lt.train()


if __name__ == "__main__":
    main()
