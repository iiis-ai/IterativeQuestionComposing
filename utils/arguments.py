import dataclasses
from dataclasses import dataclass, field

from typing import Optional, List, Union

# from peft import TaskType
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING
import transformers
from transformers.hf_argparser import DataClass
from transformers.training_args import OptimizerNames

from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
                    + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    flash_attn: bool = field(
        default=False,
        metadata={"help": ("will use flash_attn")}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    prompt_templater: str = field(
        default="mathinstruct",
        metadata={
            "help": (
                "The prompt templater to use. "
                "we now have math, mathmix, math-dataset, mathinstruct and redpajama."
            ),
        },
    )
    prompt_template_name: str = field(
        default="mmiqc",
        metadata={
            "help": (
                "The name of the prompt template to use. "
                "If not specified, the default prompt template for the model will be used."
            )
        },
    )
    prompt_template_verbose: bool = field(
        default=False,
        metadata={
            "help": ("Whether to print the prompt template before generating prompts.")
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
                self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this, "
                "-1 for all "
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this, "
                "-1 for all. "
            )
        },
    )
    train_mode: Optional[str] = field(
        default="mix",
        metadata={"help": "mode for mask", "choices": ["mix", "concat", "ignore", "wo_ignore"]},
    )
    ignore_instruction: bool = field(default=False, metadata={"help": "set instruction part to -100"})
    concat_all: bool = field(default=False, metadata={"help": "concat all data, separate by <eos> <bos>"})
    use_overflow: bool = field(default=False, metadata={"help": "Enable overflow token"})
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "The datasets processed stored"}
    )


    load_from_disk: bool = field(default=False, metadata={"help": "Load from disk"})

    num_proc: Optional[int] = field(
        default=1,
        metadata={"help": "Number of processes to use for preprocessing"},
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length"},
    )
    wandb_project: Optional[str] = field(
        default="", metadata={"help": "Wandb project name"}
    )
    wandb_name: Optional[str] = field(
        default="", metadata={"help": "Wandb run name"}
    )
    wandb_watch: Optional[str] = field(
        default="gradients", metadata={"help": "Wandb watch type"}
    )
    wandb_log_model: Optional[str] = field(
        default=False, metadata={"help": "Wandb log model"}
    )

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0",
                "The streaming feature requires `datasets>=2.0.0`",
            )


@dataclass
class PeftArguments:
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )
    peft_task_type: str = field(
        default="CAUSAL_LM", metadata={"help": "Task type"}
    )

    modules_to_save: Optional[str] = field(default=None)
    debug_mode: Optional[bool] = field(default=False)
    peft_path: Optional[str] = field(default=None)


class TrainingArguments(transformers.TrainingArguments):
    optim: Union[OptimizerNames, str] = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )


def print_args(*args_list):
    out_list = []
    for args in args_list:
        out_list.append(f"********* ARGS: {type(args).__name__} *********")
        for _field in dataclasses.fields(args):
            out_list.append(f"{_field.name}[{_field.type.__name__}]: {getattr(args, _field.name)}")
    print("\n".join(out_list))
    with open("latest_args.txt", "w") as f_out:
        f_out.write("\n".join(out_list))
