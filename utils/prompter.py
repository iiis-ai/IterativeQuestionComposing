import json

class BasePrompter(object):
    def __init__(self, template_name: str, verbose: bool = False):
        self.template = json.load(open(f"templates/{template_name}.json"))
        self._verbose = verbose

    def generate_prompt(self, data_point) -> str:
        raise NotImplementedError

    def generate_wo_response_prompt(self, data_point) -> str:
        raise NotImplementedError

    def generate_instruction(self, data_point) -> str:
        raise NotImplementedError

    def generate_response(self, data_point) -> str:
        raise NotImplementedError

    def get_response(self, data_point) -> str:
        raise NotImplementedError

class MathInstructPrompter(BasePrompter):
    # fields: instruction, output
    def __init__(
        self,
        template_name: str = "mammoth_cot",
        verbose: bool = False,
    ):
        super().__init__(template_name, verbose)
    
    def generate_prompt(self, data_point) -> str:
        res = self.template["prompt"].format(
            instruction=data_point["instruction"],
            response=data_point["output"]
        )
        if self._verbose:
            print(res)
        return res

    def generate_instruction(self, data_point) -> str:
        res = self.template["instruction"].format(
            instruction=data_point["instruction"]
        )
        if self._verbose:
            print(res)
        return res

    def generate_response(self, data_point) -> str:
        res = self.template["response"].format(
            response=data_point["output"]
        )
        if self._verbose:
            print(res)
        return res

    def generate_wo_response_prompt(self, data_point) -> str:
        res = self.template["prompt_wo_response"].format(
            instruction=data_point["instruction"]
        )
        if self._verbose:
            print(res)
        return res


PROMPTER_DICT = {
    "mathinstruct": MathInstructPrompter
}
