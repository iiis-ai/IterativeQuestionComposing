from os.path import join
import os
import json
from typing import List, Optional, Union

import pandas as pd
import fire
import torch
from vllm import LLM, SamplingParams, RequestOutput

from utils.normalize_answer import compare_modelanswer_with_answer, extract_math_answer


def eval_MATH(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    output_root: str = 'output',
    output_name: str = 'default',
    test_file: str = 'testsets/MATH-test.jsonl',
    stop: Optional[Union[str, List[str]]] = None,
    max_new_tokens: int = 2048,
):
    os.makedirs(output_root, exist_ok=True)
    output_fn = join(output_root, f'{output_name}.jsonl')

    num_gpus = torch.cuda.device_count()
    if not tokenizer_name:
        tokenizer_name = model_name
    model = LLM(model_name, tokenizer_name, trust_remote_code=True, tensor_parallel_size=num_gpus)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop=stop)
    tokenizer = model.get_tokenizer()

    if not tokenizer.eos_token_id:
        try:
            tokenizer.eos_token_id = tokenizer.eod_id
            print('Now setting eos_token_id to eod_id for Qwen models')
        except Exception as e:
            raise(f'No "eos_token_id" or "eod_id" for the tokenizer. Please specify one.')

    with open(test_file, 'r') as f:
        # has answer, problem and solution fields
        data_points = [json.loads(line) for line in f]
    
    num_correct, current_total = 0, 0
    try:
        problems = [dp['problem'] for dp in data_points]
        answers = [dp['answer'] for dp in data_points]
        solutions = [dp['solution'] for dp in data_points]
        prompts = [f'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{problem}\n\n' for problem in problems]
        
        outputs = model.generate(prompts, sampling_params) # type: RequestOutput
        output_texts = [output.outputs[0].text for output in outputs]
        num_correct, current_total = 0, 0
        for problem, answer, solution, model_solution in zip(problems, answers, solutions, output_texts):
            model_answer = extract_math_answer(model_solution)
            correct = compare_modelanswer_with_answer(answer, model_answer)
            current_total += 1
            num_correct += correct
            data_point = {
                'correct': correct, 'answer': answer, 'model_answer': model_answer,
                'problem': problem, 'solution': solution, 'model_solution': model_solution 
            }
            with open(output_fn, 'a') as f:
                f.write(json.dumps(data_point)+'\n')
    except Exception as e:
        print(f'Exception correct: {correct}')
        print(f'Exception Model Solution:{model_solution}')
        print(f'Exception Model Answer:{model_answer}')
        print(f'Encountered exception {e} during evaluation.')

    message = f'{num_correct/current_total:.4f}, {num_correct}/{current_total}, {output_fn}'
    print(message)

    return



def create_prompt(row, prompt_type='few_shot'):
    if prompt_type == 'few_shot':
        template = """Problem:
Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.

Solution:
To determine the domain, we must ensure that:
1. The expressions inside each square root are non-negative.
2. The denominator is not equal to zero.

For the numerator, $x-2 \ge 0$ gives $x \ge 2$.

For the denominator, $5-x \ge 0$ gives $x \le 5$. And since the denominator cannot be zero, $5-x > 0$ which further narrows it to $x < 5$.

Combining these results, the domain of the expression is $[2,5)$.

Final Answer: The final answer is $[2,5)$.

Problem:
If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12$, then find $\det (\mathbf{A} \mathbf{B})$.

Solution:
Using the property of determinants, we can say that:
$\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B})$.
Plugging in the given values:
$\det (\mathbf{A} \mathbf{B}) = 2 \times 12 = 24$.

Final Answer: The final answer is $24$.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?

Solution:
First, calculate the total weight Terrell lifts with the 20-pound weights:
$2 \times 12 \times 20 = 480$ pounds.
If he uses 15-pound weights and lifts them $n$ times:
$2 \times 15 \times n = 30n$ pounds.
To find $n$, set these two equal:
\begin{align*}
30n &= 480 \\
n &= \frac{480}{30} \\
n &= 16
\end{align*}

Final Answer: The final answer is $16$.

Problem:
If the system of equations
\begin{align*}
6x-4y &= a, \\
6y-9x &= b.
\end{align*}
has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\frac{a}{b}$, assuming $b$ is nonzero.

Solution:
Multiply the first equation by $-\frac{3}{2}$ to obtain:
$6y-9x = -\frac{3}{2}a$.
Since we also know that $6y-9x = b$, equating them gives:
$-\frac{3}{2}a = b$ which implies $\frac{a}{b} = -\frac{2}{3}$.

Final Answer: The final answer is $-\frac{2}{3}$."""
        template += f"\n\nProblem:\n{row['question']}\n\nSolution:\n"
    elif prompt_type == 'mammoth':
        template = f"Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{row['question']}\n\n### Response:"
    elif prompt_type == 'open_chat':
        template = f"GPT4 Correct User: {row['question']}<|end_of_turn|>GPT4 Correct Assistant:"
    elif prompt_type == 'direct':
        template = f"Answer the following question:\n{row['question']}"
    elif prompt_type == 'mmiqc':
        template = f'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{row["question"]}\n\n'
    return template

def run_exam(
    model_name,
    output_name,
    output_root = 'output',
    exam_path = 'datasets/hungarian.csv',
    tokenizer_name = None,
    prompt_type = 'few_shot'
):
    # Load the csv
    
    df = pd.read_csv(exam_path)
    # Name the columns
    df.columns = ['question']

    # Add prompts column
    df['prompt'] = df.apply(lambda row: create_prompt(row, prompt_type), axis=1)

    print(df.head())

    # Load the model    
    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=1024, stop=['\n\nProblem:'])
    num_gpus = torch.cuda.device_count()
    if not tokenizer_name:
        tokenizer_name = model_name
    llm = LLM(model_name, tokenizer_name, tensor_parallel_size=num_gpus, trust_remote_code=True)

    outputs = llm.generate(df['prompt'].tolist(), sampling_params)
    answers = [output.outputs[0].text for output in outputs]

    # Add answers column
    df['answer'] = answers

    # Save the csv
    csv_path = join(output_root, output_name) + '.csv'
    jsonl_path = join(output_root, output_name) + '.jsonl'
    df.to_csv(csv_path, index=False)
    questions = df['question']
    items = [json.dumps({'answer': answers[i], 'question': questions[i]})  
             for i in range(len(answers))]
    with open(jsonl_path, 'w', encoding='UTF-8') as f:
        f.write('\n'.join(items))


def main(task, **kwargs):
    globals()[task](**kwargs)

if __name__ == "__main__":
    fire.Fire(main)
