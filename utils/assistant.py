import os
from os.path import join
import concurrent
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict, replace
import string
import json
import requests
import random
import warnings
from time import sleep
from functools import partial

from tqdm import tqdm

from utils.normalize_answer import extract_math_answer, compare_modelanswer_with_answer

# can use asdict() function to transform dataclass instance into a dict
@dataclass
class RequestParameters:
    model: str = "gpt-4-1106-preview"
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0
    stream: bool = False


# The assistant for Azure GPT service
class Assistant():

    def __init__(self, api_key: str, azure_url: str, system_prompt: str = "") -> None:
        self.messages = [] # current messages
        self.parameters = RequestParameters()
        self.api_key = api_key
        self.timeout_length = 10
        self.uuid = self.generate_uuid()
        self.last_err_response = None
        self.model_url = azure_url
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
    
    def set_request_parameters(self, **kwargs):
        self.parameters = replace(self.parameters, **kwargs)

    # @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(30))
    def send_request(self, add_to_msg: bool = True, wait_time=10, max_retry=30, timeout=180) -> str:
        if len(self.messages) > 0:
            last_role = self.messages[-1]["role"]
            if last_role == "assistant":
                warnings.warn("Last message was from assistant as well.")
        url = self.model_url
        request = asdict(self.parameters)
        request["messages"] = self.messages
        body = {"type": "RequestAzureMessage", "apikey": self.api_key, "request": request, "uuid": {"id": self.uuid}}
        for i in range(max_retry):
            try:
                response_obj = requests.post(url, json=body, timeout=timeout)
            except Exception as e:
                print(f"{i}th try request, exception: {e}")
                if i == max_retry - 1:
                    raise Exception(f"Retry times exceed {max_retry} times, raise exception.")
                sleep(wait_time)
                continue
            if response_obj.status_code == 200:
                response = response_obj.json()
                response_message = response['choices'][0]['message']
                break
            else:
                self.last_err_response = response_obj
                text = response_obj.text
                print(f"{i}th try request error with status code {response_obj.status_code} and text: {text}")
                if '400' in text:
                    warnings.warn("Since this is a 400 error, we finish the dialogue and do not raise exception.")
                    response_message = {'role': 'assistant', 'content': '\\boxed{' + text + '}'}
                    break
                if i == max_retry - 1:
                    raise Exception(f"Retry times exceed {max_retry} times, raise exception.")
                sleep(wait_time)

        if add_to_msg:
            self.messages.append(response_message)

        try:
            content = response_message['content']
        except Exception as e:
            warnings.warn(f"Exception {e} encountered when extract content from the response\n{response_message}")
            content = ""
        return content
    
    def receive_user_prompt(self, user_prompt: str):
        message = {"role": "user", "content": user_prompt}
        self.messages.append(message)
        return

    @staticmethod
    def generate_uuid(length=10):
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))
    
    def reset_dialogue(self, remain_system_prompt=True):
        if self.messages and remain_system_prompt and self.messages[0]["role"] == 'system':
            self.messages = [self.messages[0]]
        else:
            self.messages = []

def _generate_problem_1batch(batch, api_key, sys_prompt, model, max_tokens, output_fp, temperature, add_sol, add_ans):
    assistant = Assistant(api_key=api_key, system_prompt=sys_prompt)
    assistant.set_request_parameters(max_tokens=max_tokens, model=model, temperature=temperature)
    items = [json.loads(text) for text in batch]
    for item in items:
        if not add_sol:
            del item['solution']
        if not add_ans:
            del item['answer']
    user_prompt = '\n'.join([json.dumps(item) for item in items])
    assistant.receive_user_prompt(user_prompt)
    output = assistant.send_request(add_to_msg=True)
    generated_problems = output.strip().split('\n')

    valid = 0
    with open(output_fp, 'a') as f:
        for problem_json in generated_problems:
            try:
                item = json.loads(problem_json)
                # keys = item.keys()
                assert type(item['problem']) == str
                if add_sol:
                    assert type(item['solution']) == str
                if add_ans:
                    assert type(item['answer']) == str
                f.write(problem_json+'\n')
                valid += 1
            except Exception as e:
                print(f'Exception {e} occured when decode generated problem json:\n{problem_json}')

    return valid

def generate_problem(
    api_key: str,
    example_fp: str = 'datasets/MATH-train-wo_asy.jsonl',
    model: str = 'gpt-4-1106-preview',
    output_root: str = 'output',
    output_name: str = 'genq-debug',
    num_example: int = 1,
    num_generate: int = 5,
    temperature: float = 0.0,
    sys_prompt_fp: str = 'prompts/qb.md',
    max_tokens: int = 4000,
    start: int = 0,
    end = None,
    num_worker: int = 4,
    add_sol: bool = False,
    add_ans: bool = True
):
    os.makedirs(output_root, exist_ok=True)
    output_fp = join(output_root, f'{output_name}.jsonl')
    with open(sys_prompt_fp) as f:
        sys_prompt = f.read().format(num_example=num_example, num_generate=num_generate)
    
    with open(example_fp) as f:
        texts = f.read().strip().split('\n')    # better use for line in f:
    texts = texts[start:end]
    batched_texts = [texts[k:k+num_example] for k in range(0, len(texts), num_example)]
    if len(batched_texts[-1]) != num_example:
        batched_texts = batched_texts[:-1]

    generate_problem_1batch = partial(_generate_problem_1batch, api_key=api_key, sys_prompt=sys_prompt, model=model, max_tokens=max_tokens, output_fp=output_fp, temperature=temperature, add_sol=add_sol, add_ans=add_ans)

    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        max_valid = num_generated = 0
        futures = [executor.submit(generate_problem_1batch, batch_text) for batch_text in batched_texts]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(batched_texts)):
            valid = future.result()  # Obtain the result from the completed future
            max_valid += num_generate
            num_generated += valid
            print(f'{num_generated}/{max_valid} valid problems generated so far, output to {output_fp}')

    return



def _genq_1q1a(text, api_key, sys_prompt, model, max_tokens, output_fp, temperature, add_sol, add_ans):
    assistant = Assistant(api_key=api_key, system_prompt=sys_prompt)
    assistant.set_request_parameters(max_tokens=max_tokens, model=model, temperature=temperature)
    item = json.loads(text)
    if not add_sol:
        del item['solution']
    if not add_ans:
        del item['answer']
    user_prompt = json.dumps(item)
    assistant.receive_user_prompt(user_prompt)
    output = assistant.send_request(add_to_msg=True)
    problem_json = output.strip()

    valid = 0
    with open(output_fp, 'a') as f:
        try:
            item = json.loads(problem_json)
            # keys = item.keys()
            assert type(item['problem']) == str
            if add_sol:
                assert type(item['solution']) == str
            if add_ans:
                assert type(item['answer']) == str
            f.write(problem_json+'\n')
            valid += 1
        except Exception as e:
            print(f'Exception {e} occured when decode generated problem json:\n{problem_json}')

    return valid


def genq_1q1a(
    api_key: str,
    output_name: str,
    example_fp: str = 'datasets/MATH-train-wo_asy.jsonl',
    model: str = 'gpt-4-1106-preview', 
    temperature: float = 0.0, 
    max_tokens: int = 4000,
    output_root: str = 'output', 
    sys_prompt_fp: str = 'prompts/compose_init.md',
    start: int = 0, end = None,
    num_worker: int = 1,
    add_sol: bool = True, 
    add_ans: bool = True
):
    os.makedirs(output_root, exist_ok=True)
    output_fp = join(output_root, f'{output_name}.jsonl')
    with open(sys_prompt_fp) as f:
        sys_prompt = f.read()
    
    with open(example_fp) as f:
        texts = f.read().strip().split('\n')    # better use for line in f:
    texts = texts[start:end]
    
    genq_1q1a_1 = partial(_genq_1q1a, api_key=api_key, sys_prompt=sys_prompt, model=model, max_tokens=max_tokens, output_fp=output_fp, temperature=temperature, add_sol=add_sol, add_ans=add_ans)

    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        max_valid = num_generated = 0
        futures = [executor.submit(genq_1q1a_1, text) for text in texts]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(texts)):
            valid = future.result()  # Obtain the result from the completed future
            max_valid += 1
            num_generated += valid
            print(f'{num_generated}/{max_valid} valid problems generated so far, output to {output_fp}')

    return


def _reject_sample_1q(item, api_key, sys_prompt, model, max_tokens, output_fp, temperature, num_sample_1q, max_valid_1q):

    assistant = Assistant(api_key=api_key, system_prompt=sys_prompt)
    assistant.set_request_parameters(max_tokens=max_tokens, model=model, temperature=temperature)
    user_prompt = item['problem']
    answer = item['answer']

    new_items = []
    for _ in range(num_sample_1q):
        assistant.reset_dialogue(remain_system_prompt=True)
        assistant.receive_user_prompt(user_prompt)
        output = assistant.send_request(add_to_msg=True) # type: str
        model_answer = extract_math_answer(output)
        if compare_modelanswer_with_answer(answer, model_answer):
            new_items.append({'problem': item['problem'], 'solution': output, 'answer': item['answer'], 'source': model})
        if len(new_items) >= max_valid_1q:
            break

    num_generated = len(new_items)
    with open(output_fp, 'a') as f:
        for new_item in new_items:
            f.write(json.dumps(new_item)+'\n')

    return num_generated


def reject_sample(
    api_key: str,
    question_fp: str = 'datasets/MATH-train-wo_asy.jsonl',
    model: str = 'gpt-3.5-turbo',
    output_root: str = 'output',
    output_name: str = 'rej_sample-debug',
    temperature: float = 0.0,
    sys_prompt_fp: str = 'prompts/cot.md',
    max_tokens: int = 4000,
    start: int = 0,
    end = None,
    num_sample_1q: int = 3,
    max_valid_1q: int = 3,
    num_worker: int = 4
):
    # the problem file shall include "problem" and "answer"
    os.makedirs(output_root, exist_ok=True)
    output_fp = join(output_root, f'{output_name}.jsonl')
    with open(sys_prompt_fp) as f:
        sys_prompt = f.read()
    
    items = []
    with open(question_fp) as f:
        for line in f:
            if line.strip() == '':
                continue
            items.append(json.loads(line.strip()))
    items = items[start:end]

    reject_sample_1q = partial(_reject_sample_1q, api_key=api_key, sys_prompt=sys_prompt, model=model, max_tokens=max_tokens, output_fp=output_fp, temperature=temperature, num_sample_1q=num_sample_1q, max_valid_1q=max_valid_1q)

    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        max_valid = 0
        num_generated = 0

        futures = {executor.submit(reject_sample_1q, item) for item in items}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(items)):
            num_gen_1q = future.result()
            max_valid += max_valid_1q
            num_generated += num_gen_1q
            print(f'{num_generated}/{max_valid} valid solutions generated so far, output to {output_fp}')

    return
