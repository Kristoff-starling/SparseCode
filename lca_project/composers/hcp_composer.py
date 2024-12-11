from composers.one_completion_file_composer import OneCompletionFileComposer
from data_classes.datapoint_base import DatapointBase
import os
from pathlib import Path
import shutil
import sys
sys.path.append(os.path.join(Path(__file__).parent.parent.parent, "HCP-Coder"))
from composer_entry import ComposerEntry

class HCPComposer(OneCompletionFileComposer):

    def repo_dir(self, datapoint: DatapointBase) -> str:
        return os.path.join(str(Path(__file__).parent.parent), "data", "tmp", datapoint.repo_name)

    def expand_repo(self, datapoint: DatapointBase, line_num: int):
        base_dir = self.repo_dir(datapoint)
        context = datapoint.get_context()
        for path, contents in context.items():
            full_path = os.path.join(base_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(contents)
        completion = datapoint.get_prefix(line_num)
        completion_path = os.path.join(base_dir, datapoint.get_completion_file())
        os.makedirs(os.path.dirname(completion_path), exist_ok=True)
        with open(completion_path, "w") as f:
                f.write(completion)

    def cleanup_repo(self, datapoint: DatapointBase):
        base_dir = self.repo_dir(datapoint)
        shutil.rmtree(base_dir)        

    def context_composer(self, datapoint: DatapointBase) -> dict[int, str]:
        per_line_context = dict()
        for _, lines in list(datapoint.completion_lines.items()):
             for line_num in lines:  
                self.expand_repo(datapoint, line_num)

                base_dir = self.repo_dir(datapoint)
                completion_path = os.path.join(base_dir, datapoint.get_completion_file())
                try:
                    hcp_result_dict = ComposerEntry.run_hcp(base_dir, completion_path, line_num-1, 5, 0.8)
                    hcp_result = list(hcp_result_dict.values())[0]
                    composed_content = [path + self.meta_info_sep_symbol + content for path, content in hcp_result.items() if not(content.strip() == '')]
                    print(sum([len(x) for x in composed_content]))
                except:
                    composed_content = []

                repo_name = datapoint.repo_name

                completion = datapoint.get_completion()
                assert len(completion) == 1, 'Only one file should be completed'
                completion_path = list(completion)[0]
                composed_content.append(completion_path + self.meta_info_sep_symbol)

                repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"

                final = repo_metainfo + self.lang_sep_symbol.join(composed_content)

                self.cleanup_repo(datapoint)

                per_line_context[line_num] = final

        #for k in list(per_line_context.keys()):
        #    print(k)
        #    print(per_line_context[k][:50])

        return per_line_context
