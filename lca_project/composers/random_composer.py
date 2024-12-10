from composers.one_completion_file_composer import OneCompletionFileComposer
from data_classes.datapoint_base import DatapointBase
import random


class RandomComposer(OneCompletionFileComposer):
    def context_composer(self, datapoint: DatapointBase) -> str:
        # context = datapoint['context']
        context = datapoint.get_context()
        keys = list(context.keys())
        random.shuffle(keys)
        composed_content = [path + self.meta_info_sep_symbol + context[path] for path in keys]

        return super().compose_with_context_truncation(composed_content, datapoint)
