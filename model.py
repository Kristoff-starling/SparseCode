import torch
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from lca_project.model_hub.llama_sparse_attention import LlamaSparseForCausalLM

from utils import time_monitor


class StopOnEndOfQueryCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_token="<end_of_query>"):
        self.stop_token_ids = tokenizer.encode(stop_token, add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) >= len(self.stop_token_ids):
            return input_ids[0][-len(self.stop_token_ids):].tolist() == self.stop_token_ids
        return False


def generate_prompt(context: str) -> str:
    prompt_template = """Please generate the incomplete function. 
<start_of_context>
{context}
<end_of_context>

<start_of_query>
def sum(x, y):
"""
    prompt = prompt_template.format(context=context)
    return prompt


device = "cuda:0"
sparse_deepseek_model = LlamaSparseForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", attn_implementation="eager")
sparse_deepseek_model.to(device)
deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")


@time_monitor
def generate_response(prompt: str) -> str:
    inputs = deepseek_tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.cuda()
    output_ids = sparse_deepseek_model.generate(
        input_ids=input_ids,
        stopping_criteria=StoppingCriteriaList([StopOnEndOfQueryCriteria(deepseek_tokenizer)]),
        max_new_tokens=1000,
    )
    output_text = deepseek_tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )
    # extract answer
    output_text = output_text.split("<start_of_query>")[-1].split("<end_of_query>")[0].strip()
    return output_text


context=""
prompt = generate_prompt(context)
print(generate_response(prompt))
