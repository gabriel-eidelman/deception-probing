import json
from evaluate_test_set import evaluate_test_set

prompts = json.load(open("hypothesis_test/distribution_fix_hypothesis_prompts.json"))["prompts"]
evaluate_test_set(test_set=prompts, output_path="hypothesis_results.json")