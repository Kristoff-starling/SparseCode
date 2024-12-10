from data_classes.datapoint_base import DatapointBase


class ComposerBase:
    def __init__(self,
                 lang_sep_symbol,
                 meta_info_sep_symbol,
                 extension,
                 seq_max_len
                 ):
        self.lang_sep_symbol = lang_sep_symbol
        self.meta_info_sep_symbol = meta_info_sep_symbol
        self.extension = extension
        self.seq_max_len = seq_max_len

    def context_composer(self, datapoint: DatapointBase) -> str:
        raise NotImplementedError

    def completion_composer(self, datapoint: DatapointBase) -> str:
        raise NotImplementedError

