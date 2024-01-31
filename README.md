# Evaluate Models

To evaluate the models on MATH, run
```bash
python eval.py --task eval_MATH --model_name $MODEL_PATH --output_name $OUTPUT_NAME --test_file $MATH_TEST_FILE_PATH
```

To run hungarian test, run
```bash
python eval.py --task run_exam --model_name $MODEL_PATH --output_name $OUTPUT_NAME
```

# Generate MMIQC

We access GPTs through in Azure OpenAI Service. To start, add the API key to the environment variables:
```bash
export API_KEY=[Your Azure API key]
``` 

The synthetic part of MMIQC can be created by:

```bash
# Answer Augmentation
python main.py --task reject_sample --api_key $API_KEY --question_fp datasets/MATH-train-wo_asy.jsonl --output_name AnsAug

# Question Bootstrapping
python main.py --task generate_problem --api_key $API_KEY --sys_prompt_fp prompts/qb.md --output_name QB_question --num_example 1 --num_generate 5 && \
python main.py --task reject_sample --api_key $API_KEY --question_fp output/QB_question.jsonl --output_name QB_rej_sample

# Augmented Similar Problems
python main.py --task generate_problem --api_key $API_KEY --sys_prompt_fp prompts/similar.md --output_name similar_question --num_example 1 --num_generate 3 --add_sol True && \
python main.py --task reject_sample --api_key $API_KEY --question_fp output/similar_question.jsonl --output_name similar_rej_sample

# IQC iter #0
python main.py --task genq_1q1a --api_key $API_KEY --sys_prompt_fp prompts/compose_init.md --output_name iqc_iter0 && \
python main.py --task reject_sample --api_key $API_KEY --question_fp output/iqc_iter0.jsonl --output_name iqc_iter0_rej_sample

# IQC Iter #1
python main.py --task genq_1q1a --api_key $API_KEY --sys_prompt_fp prompts/compose_iter.md --output_name iqc_iter1 --example_fp output/iqc_iter0.jsonl && \
python main.py --task reject_sample --api_key $API_KEY --question_fp output/iqc_iter1.jsonl --output_name iqc_iter1_rej_sample

# ...

```

You can download the MMIQC dataset from [this link](https://www.sendbig.com/view-files/?Id=ee661a39-7d7c-4666-66fe-b24f51a65654).


# Fine-tune

To fine-tune Mistral-7B on MMIQC on 8 x 80G A800 gpus, run 
```bash
export DS_PATH=[path_to_mmiqc]; \
export OUTPUT_DIR=[output directory of fine-tuned model]; \
export BASE_MODEL="mistralai/Mistral-7B-v0.1"; \
torchrun --nproc_per_node 8 --master_port 12136 pretrain.py \
  --deepspeed ds_config/zero3.json \
  --model_name_or_path $BASE_MODEL \
  --tokenizer_name_or_path $BASE_MODEL \
  --dataset_dir $DS_PATH \
  --output_dir $OUTPUT_DIR \
  --bf16 True --tf32 True \
  --flash_attn \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type linear \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 \
  --save_total_limit 8 --logging_steps 10  \
  --save_strategy "steps" --save_steps 500 \
  --gradient_checkpointing True \
  --max_seq_length 2048 --group_by_length --num_proc 16
```
