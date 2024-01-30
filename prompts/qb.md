You will be provided with {num_example} math problem in newline-delimited json format. Please augment {num_generate} diverse problems from the given problem.

The way you augment a problem can be:
- Rephrase the problem.
- Change the scenario without modifying specific quantities.
- Set 1 number in the problem to an unknown variable, put the answer in the problem and ask what is the value of the variable. Ensure the generated problem is reasonable. Otherwise, skip this method.
- Other approaches that can ensure the correctness of the answer you provide to the augmented problem.

Your response should only contain text in newline-delimited json format, keeping the same with the given problem. Please use two backslashes to represent one in the strings.