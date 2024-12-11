import json
import os
import shutil
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union
import sys
import datetime

import hydra
import omegaconf
import torch.cuda
# import wandb
from omegaconf import DictConfig

from composers.composer_registry import COMPOSERS
from eval.eval import evaluate
from eval.line_generators import evaluate_generation, GeneratorConfig
from eval.preprocess import preprocess
from model_hub.model_inference import inference
from model_hub.model_registry import MODEL_REGISTRY


@dataclass
class PreprocessConfig:
    model: str  # One of PREPROCESSORS from lca.code_generation.eval.preprocess
    dataset: Union[str, omegaconf.dictconfig.DictConfig]  # Path to dataset or dictionary with `path`, `name` keys
    tokenizer: str  # Path to tokenizer
    # config_path: str  # Path to composer configs
    composers: str  # One of COMPOSERS from lca.code_generation.eval.preprocess
    out_dir: str  # Where to save preprocessed dataset
    context_len_char: int  # How much do we need to crop context string 5*seq_max_len by default


@dataclass
class InferenceConfig:
    model: str  # One of MODEL_REGISTRY from model_hub.model_inference
    mode: str # dense or sparse attention
    attention_visualization: bool # visualize attention weights
    input_data_path: str  # the same as PreprocessConfig.out_dir
    seq_max_len: int  # Maximal possible sequence length
    context_max: int  # Maximal possible context length
    out_dir: str  # Directory to save logits


@dataclass
class EvalConfig:
    device: str
    out_dir: str  # Directory to save results of the evaluation
    dataset_dir: str  # the same as InferenceConfig.out_dir


class EvalPipeline:
    def __init__(self, config, composers=COMPOSERS):
        self.config = config

        preprocess_params = config.params.preprocess_params
        inference_params = config.params.inference_params
        eval_params = config.params.eval_params
        wandb_project_name = config.wandb_project_name

        self.config.composers_config['seq_max_len'] = inference_params['seq_max_len']

        assert inference_params['model'] in MODEL_REGISTRY, (f'config: inference_params: model: '
                                                             f'{inference_params["model"]} is not in MODEL_REGISTRY')
        if MODEL_REGISTRY[inference_params['model']].checkpoint != preprocess_params['tokenizer']:
            warnings.warn(f'Model and Tokenizer have different paths')

        # preprocess_params.dataset = config.dataset
        if isinstance(config.dataset, str):
            self.dataset_name = config.dataset.split('/')[-1].replace('-', '_')
        elif isinstance(config.dataset, omegaconf.dictconfig.DictConfig):
            self.dataset_name = config.dataset['name']
        dataset_out_dir = os.path.join(config.artifacts_dir, config.language, inference_params['model'],
                                       self.dataset_name)
        # preprocess_params['out_dir'] = os.path.join(dataset_out_dir, 'in')
        # inference_params['out_dir'] = os.path.join(dataset_out_dir, 'out')
        # eval_params['out_dir'] = os.path.join(dataset_out_dir, 'results')

        # eval_params["dataset_dir"] = inference_params["out_dir"]
        self.preprocess_args = PreprocessConfig(dataset=config.dataset, out_dir=os.path.join(dataset_out_dir, 'in'),
                                                context_len_char=5 * inference_params['seq_max_len'],
                                                **preprocess_params)
        self.inference_args = InferenceConfig(out_dir=os.path.join(dataset_out_dir, 'out'), 
                                              input_data_path=os.path.join(os.getcwd(), dataset_out_dir, 'in', 'model_inputs_composer_' + self.preprocess_args.composers + '.json'), 
                                              **inference_params)
        self.eval_args = EvalConfig(dataset_dir=self.inference_args.out_dir,
                                    out_dir=os.path.join(dataset_out_dir, 'results'), **eval_params)
        self.out_dir = os.path.join(dataset_out_dir, 'results')
        self.composers = {self.preprocess_args.composers: composers[self.preprocess_args.composers]}
        self.project_name = wandb_project_name
        self.generator_config: GeneratorConfig

    def _resolve_directories(self):
        inference_out_dir_path = Path(self.inference_args.out_dir)
        if inference_out_dir_path.exists():
            shutil.rmtree(inference_out_dir_path)
        inference_out_dir_path.mkdir(parents=True, exist_ok=True)

        eval_out_dir_path = Path(self.eval_args.out_dir)
        eval_out_dir_path.mkdir(parents=True, exist_ok=True)

    def run(self):
        do_generation = self.config.do_generation
        seed = self.config.seed
        results = list()

        # Initialization of config for LineGenerator
        self.generator_config = GeneratorConfig(
            input_data_path=self.inference_args.input_data_path,
            seq_max_len=self.inference_args.seq_max_len - 100,
            context_max=self.inference_args.context_max,
            model=self.inference_args.model,
            mode=self.inference_args.mode,
            attention_visualization=self.inference_args.attention_visualization,
            device=self.eval_args.device,
            best_perplexity=sys.maxsize,
            tokenizer_path=self.preprocess_args.tokenizer,
            composer=self.preprocess_args.composers,
            seed=seed,
            results_path=os.path.join(self.out_dir, 'generation_results.jsonl')
        )
        # manual - create results directory
        if not os.path.exists(os.path.join(os.getcwd(), self.out_dir)):
            os.makedirs(os.path.join(os.getcwd(), self.out_dir))

        for composer in self.composers:
            if composer == 'none':
                continue
            print('running composer')
            results = self.run_composer(composer, results)
            print('finished composer')

        inference_out_dir_path = Path(self.inference_args.out_dir)
        if inference_out_dir_path.exists():
            shutil.rmtree(inference_out_dir_path)

        if do_generation:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # wb_run = wandb.init(
            #     project=self.config.wandb_project_name_generation,
            #     name='_'.join([self.generator_config.model, 'composer', self.generator_config.composer, self.dataset_name]),
            #     config=asdict(self.generator_config)
            # )
            gen_scores, gen_results, em_difference, line_counts = evaluate_generation(self.generator_config)

            # wb_run.log(gen_scores | {'EM_difference': em_difference, 'Line Counts': line_counts,
            #                          "dataset": self.dataset_name, "model": self.inference_args.model})
            # wb_run.finish()
            score_file = f"generation_scores_{timestamp}_{self.inference_args.mode}_{'-'.join(self.composers)}.json"
            with open(os.path.join(self.out_dir, score_file), 'w') as f:
                json.dump(gen_results, f, indent=4)
            print(f">>Generation Results are in {os.path.join(self.out_dir, score_file)}")

            return results, gen_results

        return results

    def run_zero_context(self):
        self.inference_args.context_max = 0
        self.eval_args.out_dir = os.path.join(self.out_dir, "context_0")
        self._resolve_directories()

        print(">>Context 0 run")

        print('>>Preprocessing...')
        prepared_dataset_path = preprocess(self.preprocess_args, self.config.composers_config)
        # prepared_dataset_path = '/home/glukhov/long_code_arena/lca/data/h3_pretrained_in/model_inputs_composer_none.json'

        print(">>Model inference...")
        self.inference_args.input_data_path = prepared_dataset_path
        lost_tokens = inference(self.inference_args)

        print(">>Evaluation...")
        mean_ppl = evaluate(self.eval_args)

        return {"perplexity": mean_ppl, "context": 0, "composer": "zero", "dataset": self.dataset_name,
                "model": self.inference_args.model} | lost_tokens

    def run_composer(self, composer, results):
        # wb_run = wandb.init(project=self.project_name, group=f"{composer} composer", name=f"{composer} composer")
        self.preprocess_args.composers = composer
        print(f'>>Preprocessing for {composer} composer...')
        prepared_dataset_path = preprocess(self.preprocess_args, self.config.composers_config)
        self.inference_args.input_data_path = prepared_dataset_path
        
        # wb_run.finish()

        return results


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    pipeline = EvalPipeline(cfg)#cfg.preprocess_params, cfg.inference_params, cfg.eval_params,
                            # wandb_project_name=cfg.wandb_project_name)
    results = pipeline.run()
    print()
    print(results)


if __name__ == '__main__':
    main()

    # preprocess_params = {
    #     "model": "huggingface",
    #     "dataset": "/home/glukhov/long_code_arena/lca/data/python/benchmark_data_538_1239_sample_100.json", #"/home/glukhov/long_code_arena/lca/data/kotlin/benchmark_data_815_1846_sample_100.json", #"/home/glukhov/long_code_arena/lca/data/python/benchmark_data_538_1239_sample_100.json",
    #     "tokenizer": "bigcode/starcoderbase-1b",  #"bigcode/starcoder", #"Salesforce/codegen25-7b-mono",
    #     "config_path": "/home/glukhov/long_code_arena/lca/lca/code_generation/eval/composer_config/config.json",
    #     "out_dir": "/home/glukhov/long_code_arena/lca/data/python/starcoder1b_4bit_in",
    # }
    #
    # inference_params = {
    #     "model": "starcoder1b",
    #     "seq_max_len": 8190,
    #     "out_dir": "/mnt/data/shared-data/lca/code_generation/data/python/starcoder1b_4bit_out",
    #     "input_data_path": "",
    #     "context_max": -1,
    # }
    #
    # project_name = "Starcoder 1B filtered data 100 Python + New composers"
    #
    # eval_params = {
    #     "device": "cuda",
    #     "out_dir": "/home/glukhov/long_code_arena/lca/data/python/starcoder1b_4bit_out",
    #     "dataset_dir": ""
    # }
    #
    #
    #
    # # preprocess_params = {
    # #     "model": "fl_python",
    # #     "dataset": "/home/glukhov/long_code_arena/lca/data/benchmark_data_filtered_250.json",
    # #     "tokenizer": "gpt2",  #"/home/glukhov/fl-safari/fl-pipeline/out_dirs_virtual/out_dir_starcoder_python/data_processors/gpt2_pretrained", #"Salesforce/codegen25-7b-mono",
    # #     "config_path": "/home/glukhov/long_code_arena/lca/lca/code_generation/eval/config/config_fl.json",
    # #     "out_dir": "/home/glukhov/long_code_arena/lca/data/h3_pretrained_in",
    # # }
    # #
    # # inference_params = {
    # #     "model": "h3_pretrained_fl",
    # #     "seq_max_len": 8192,
    # #     "out_dir": "/mnt/data/shared-data/lca/code_generation/data/h3_pretrained_out",
    # #     "input_data_path": "",
    # #     "context_max": -1,
    # # }
    # #
    # # eval_params = {
    # #     "device": "cuda",
    # #     "out_dir": "/home/glukhov/long_code_arena/lca/data/h3_pretrained_out",
    # #     "dataset_dir": ""
    # # }
    #
    # pipeline = EvalPipeline(preprocess_params, inference_params, eval_params,
    #                         wandb_project_name=project_name)
    # results = pipeline.run()
    #
    # print(results)
