import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from model_hub.llama_sparse_attention import LlamaSparseForCausalLM


class ModelBuilderBase:
    @classmethod
    def build_model(cls, **kwargs):
        raise NotImplementedError


class HFModelBuilder(ModelBuilderBase):
    SEND_TO_DEVICE = True
    @classmethod
    def build_model(cls, checkpoint, mode="dense", **kwargs):
        kwargs = cls._update_kwargs(checkpoint, kwargs)
        device = cls._get_device()
        if mode == "dense":
            print(f"Evaluating dense model {checkpoint}")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                attn_implementation="eager",
                **kwargs
            )
        elif mode == "sparse":
            print(f"Evaluating sparse model {checkpoint}")
            model = LlamaSparseForCausalLM.from_pretrained(
                checkpoint,
                attn_implementation="eager",
                **kwargs
            )
        else:
            raise ValueError(f"Unrecognized mode {mode}")
        if cls.SEND_TO_DEVICE:
            model = model.to(device)
        model.eval()
        # model = model.to_bettertransformer()
        print('model is ready')
        return model, device

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            print('cuda is available')
            return torch.device('cuda')
        else:
            print('cuda not available')
            return torch.device('cpu')

    @staticmethod
    def _update_kwargs(checkpoint, kwargs):
        # temp: disable flash attention, couldn't download package
        #if 'attn_implementation' not in kwargs:
        #    if 'starcoder' not in checkpoint:  # Quick fix for Flash-attention 2 and starcoder
        #        kwargs['attn_implementation'] = 'flash_attention_2'
        if 'torch_dtype' not in kwargs:
            kwargs['torch_dtype'] = torch.bfloat16

        return kwargs


class HFModelBuilder4bit(HFModelBuilder):
    SEND_TO_DEVICE = False

    @classmethod
    def _update_kwargs(cls, checkpoint, kwargs):
        if 'attn_implementation' not in kwargs:
            if 'starcoder' not in checkpoint:  # Quick fix for Flash-attention 2 and starcoder
                kwargs['attn_implementation'] = 'flash_attention_2'
        if 'quantization_config' not in kwargs:
            kwargs['quantization_config'] = cls._get_q_config()

        return kwargs

    @staticmethod
    def _get_q_config():
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return q_config
