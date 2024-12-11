from src.topo import RepoTopo
from src.retriever import AutoRetriever

class ComposerEntry():

    @staticmethod
    def run_hcp(path: str, file: str, row: int, top_k: int, top_p: float) -> str:
        # Initialize the RepoTopo object
        repo_topo = RepoTopo(path)

        # define the retriever used to retrieve the related functions
        # we also support engine: 'jina'
        retriever = AutoRetriever(engine='jina')

        # get the file node object of the current file
        file_node = repo_topo.file_nodes[file]

        # get the hierarchical cross file context
        cross_file_context = repo_topo.get_hierarchical_cross_file_context(
            retriever,
            file_node,
            row=row,
            col=0,
            top_k=[top_k],
            top_p=[top_p],
        )

        return cross_file_context
