import re
from sympy import Eq, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from multiprocessing import Process, Queue

# Define a function which checks equality and puts the result in a queue.
def check_equality(expr1, expr2, queue):
    try:
        result = expr1.equals(expr2)
    except Exception as e:
        # print(f'Exception {e} occured for {expr1} vs {expr2}')
        result = False
    queue.put(result)

def equals_with_timeout(expr1, expr2, timeout=10):
    # Create a queue to share results.
    queue = Queue()

    # Run the check_equality function in a separate process.
    p = Process(target=check_equality, args=(expr1, expr2, queue))
    p.start()

    # Wait for the process to complete or timeout.
    p.join(timeout=timeout)

    if p.is_alive():
        # Terminate the process if it is still running after the timeout.
        p.terminate()
        p.join()
        print(f'sympy timeout for {expr1} vs {expr2} after {timeout} sec')
        return None  # Indicates that a timeout occurred.
    else:
        # Get the result from the queue.
        return queue.get()

def detect_answer_in_box(solution: str) -> (int, int):
    after_box = solution.split('\\boxed')[-1]
    if not after_box or after_box[0] != '{':
        raise Exception(f"No boxed latex format find in {solution}")
    stack = 1
    a = ''
    for c in after_box[1:]:
        if (c == '{'):
            stack += 1
            a += c
        elif (c == '}'):
            stack -= 1
            if (stack == 0): break
            a += c
        else:
            a += c
    ans_pos = solution.find('\\boxed') + len('\\boxed') + after_box.find(a)
    ans_len = len(a)
    return ans_pos, ans_len

def extract_from_box(pred_str):
    ans = pred_str.split('boxed')[-1]
    if not ans:
        return ""
    if (ans[0] == '{'):
        stack = 1
        a = ''
        for c in ans[1:]:
            if (c == '{'):
                stack += 1
                a += c
            elif (c == '}'):
                stack -= 1
                if (stack == 0): break
                a += c
            else:
                a += c
    else:
        a = ans.split('$')[0].strip()
    return a

def extract_math_answer(pred_str: str) -> str:
    if 'boxed' in pred_str:
        pred = extract_from_box(pred_str)
    elif('The answer is:' in pred_str):
        pred = pred_str.split('The answer is:')[-1].strip()
    elif('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    elif('the answer is ' in pred_str):
        pred = pred_str.split('the answer is ')[-1].strip()
    else:
        pattern = '-?\d*\.?\d+'
        preds = re.findall(pattern, pred_str)
        if(len(preds) >= 1):   # find the last number in the answer
            pred = preds[-1]
        else:
            pred = ""
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if 'boxed' in pred:
        pred = extract_from_box(pred_str)
    return pred.strip()


SUBSTITUTIONS = [
    ('.$', '$'), ('\\$', ''), ('\\%', ''),
    ('mbox', 'text'), (',\\text{and}', ','), ('\\text{and}', ','),
    ('\\text{m}', '\\text{}'), ('dfrac', 'frac')
]
REMOVED_EXPRESSIONS = [
    '\\left', '\\right', '\\quad', 'calories', 'calory', 'values', 'value',
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'miles', 'euros', 'euro',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'centi', 'days',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'dogs', 'boxes',
    'gallons', 'gallon', 'cubic', 'min', 'distinct', 'pieces', 'piece', 'USD', 'daps',
    'dap', 'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{ s}', '\\text{ }', '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{ }^2'
    '\\text{}^3', '\\text{ }^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
    r'^\circ', r'^{\circ}', r'\,', r'\;', r',\!', '{,}', '"', '\\dots',
    r'\ ', ' '
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    # minerva do not have this filter. it splits directly.
    parts = final_answer.split('=')
    if sum(['y' in parts[0], 'x' in parts[0], 'z' in parts[0]]) >= 2:
        pass
    else:
        final_answer = parts[-1]
    # need space to indicate whether to substitute
    final_answer = re.sub(r'([\\\w]*)/([\\\w]*)', r'\\frac{\1}{\2}', final_answer)

    # I adjusted the order
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)


    if re.match(r'\.\d+', final_answer):
        final_answer = '0' + final_answer

    # Extract answer that is in LaTeX math, is bold, is surrounded by a box, etc.
    # final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(?<!\.)\.$', '', final_answer)  # remove the dot
    final_answer = re.sub(r'(\\text\{\()(.*?)(\)\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)


    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(r'(frac)([^{])(\d)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    # \frac{a}b -> \frac{a}{b}
    # \fraca{b} -> \frac{a}{b}
    final_answer = re.sub(r'frac{(\d+)}(\d+)', r'frac{\1}{\2}', final_answer)
    final_answer = re.sub(r'frac(\d+){(\d+)}', r'frac{\1}{\2}', final_answer)

    # a\frac{b}{c} -> \frac{eval(a*c+b)}{c}
    def _repl_mixed(match: re.Match[str]) -> str:
        integer, nume, deno = [int(num) for num in match.groups()]
        new_nume = eval(f'{integer}*{deno} + {nume}')
        return f'\\frac{{{new_nume}}}{{{deno}}}'
    final_answer = re.sub(r'(\d+)\\frac{(\d+)}{(\d+)}', _repl_mixed, final_answer)
    
    # dot in $content.$ is replaced with $ at the very first
    final_answer = final_answer.replace('$', '')   

    return final_answer

def compare_modelanswer_with_answer(
    answer: str,
    model_answer: str,
    timeout: int = 10
)-> bool:
    correct = False
    model_answer = normalize_final_answer(model_answer)
    answer = normalize_final_answer(answer)
    try:
        correct = (set(model_answer.split(',')) == set(answer.split(',')))
    except Exception as e:
        print(f'An exception occurs when compare answers:\n{e}')
        correct = False
    if not correct:
        try:
            if "=" in answer and '=' in model_answer:
                # equations that = is not removed
                eq1, eq2 = parse_latex(model_answer), parse_latex(answer)
                equation1 = Eq(eq1.lhs - eq1.rhs, 0)
                equation2 = Eq(eq2.lhs - eq2.rhs, 0)

                # Simplify and compare
                correct = simplify(equation1.lhs - equation2.lhs) == 0
            elif "," not in model_answer and "," not in answer:
                model_answer_sympy = parse_latex(model_answer)
                answer_sympy = parse_latex(answer)
                # correct =  model_answer_sympy == answer_sympy 
                correct = equals_with_timeout(model_answer_sympy, answer_sympy, timeout=timeout)
                # correct =  model_answer_sympy.equals(answer_sympy)
            else:
                manss, anss = parse_expr(model_answer), parse_expr(answer)
                correct = manss == anss
            if correct:
                # print(f'New correct with sympy for {model_answer} v.s. {answer}')
                pass
        except:
            # print(f'Parse error for {model_answer} v.s. {answer}')
            pass
    if correct not in [True, False]:
        correct = False
    return correct
