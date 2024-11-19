import torch
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from utils import time_monitor


class SparseAttention(GPTNeoXAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        qkv = self.qkv(hidden_states)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim)

        seq_length = query.size(2)
        block_size = 16
        output = torch.zeros_like(query)

        for start in range(0, seq_length, block_size):
            end = min(start + block_size, seq_length)
            q_block = query[:, :, start:end, :]
            k_block = key[:, :, start:end, :]
            v_block = value[:, :, start:end, :]

            scores = torch.einsum("bhqd,bhkd->bhqk", q_block, k_block) / (self.head_dim ** 0.5)
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, start:end, start:end]

            probs = torch.nn.functional.softmax(scores, dim=-1)
            output[:, :, start:end, :] = torch.einsum("bhqk,bhvd->bhqd", probs, v_block)

        output = self._merge_heads(output, self.num_attention_heads, self.head_dim)
        return self.out_proj(output)


class SparseModel(AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # Replace the attention layers in each Transformer block
        for i, layer in enumerate(self.transformer.h):
            layer.attention = SparseAttention(config)


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
sparse_deepseek_model = SparseModel.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
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
