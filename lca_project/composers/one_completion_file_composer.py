from composers.base_composer import ComposerBase
from data_classes.datapoint_base import DatapointBase


class OneCompletionFileComposer(ComposerBase):
    def __init__(self, **composer_args):
        super().__init__(**composer_args)

    def completion_composer(self, datapoint: DatapointBase) -> str:
        completion = datapoint.completion_dict
        assert len(completion) == 1, 'Only one file should be completed'
        content = list(completion.values())[0]
        return content
    
    def repo_metainfo(self, repo_name: str) -> str:
        return f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"
    
    def compose_with_context_truncation(self, composed_content: list, datapoint: DatapointBase, truncate=False) -> str:
        completion = datapoint.get_completion()
        assert len(completion) == 1, 'Only one file should be completed'
        completion_path = list(completion)[0]

        repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{datapoint.repo_name}{self.meta_info_sep_symbol}"

        if truncate:
            # truncate composed_context until everything will fit in seq_max_len
            boilerplate_len = len(repo_metainfo) + len(completion_path) + len(self.completion_composer(datapoint))
            composed_content_lens = [len(s) for s in composed_content]
            # estimate: 2.6 characters -> 1 token
            max_len = int(self.seq_max_len * 2.6)
            i = 0
            total = sum(composed_content_lens) + boilerplate_len
            if total > max_len:
                while i < len(composed_content) and total - composed_content_lens[i] > max_len:
                    total -= composed_content_lens[i]
                    i += 1
                
                diff = max_len - total
                if i < len(composed_content):
                    composed_content = [composed_content[i][:diff]] + composed_content[i+1:]
                else:
                    composed_content = [composed_content[i-1][:diff]]

        composed_content.append(completion_path + self.meta_info_sep_symbol)

        final = repo_metainfo + self.lang_sep_symbol.join(composed_content)

        return final
