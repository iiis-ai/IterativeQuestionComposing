from utils.assistant import reject_sample, genq_1q1a, generate_problem
import fire

def main(task, **kwargs):
    globals()[task](**kwargs)

if __name__ == "__main__":
    fire.Fire(main)
