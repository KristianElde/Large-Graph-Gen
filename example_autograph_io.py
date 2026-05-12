import torch

from graph_tokenization import AutoGraphTokenizer, graph_to_tokens


def edge_index_to_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = 1
    return adj


def tokens_to_text(tokens: torch.Tensor, tokenizer: AutoGraphTokenizer) -> str:
    special = {
        tokenizer.sos: "<sos>",
        tokenizer.reset: "<reset>",
        tokenizer.ladj: "<ladj>",
        tokenizer.radj: "<radj>",
        tokenizer.eos: "<eos>",
        tokenizer.pad: "<pad>",
    }
    parts = []
    for token in tokens.tolist():
        if token in special:
            parts.append(special[token])
        else:
            parts.append(f"n{token - tokenizer.idx_offset}")
    return " ".join(parts)


def reconstruct_graph_from_tokens(
    tokens: torch.Tensor, tokenizer: AutoGraphTokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    reconstructed = tokenizer.decode(tokens)
    reconstructed_adj = edge_index_to_adjacency(
        reconstructed.edge_index, reconstructed.num_nodes
    )
    return reconstructed.edge_index, reconstructed_adj


def main() -> None:
    adjacency_matrix = torch.tensor(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=torch.long,
    )

    edge_index = adjacency_matrix.nonzero(as_tuple=False).t().contiguous()
    num_nodes = adjacency_matrix.size(0)

    tokenizer = AutoGraphTokenizer(undirected=True, append_eos=True)
    tokenizer.set_num_nodes(num_nodes)

    tokens = graph_to_tokens(edge_index=edge_index, num_nodes=num_nodes, tokenizer=tokenizer)
    reconstructed_edge_index, reconstructed_adjacency = reconstruct_graph_from_tokens(
        tokens, tokenizer
    )

    print("=== INPUT ===")
    print("Adjacency matrix:")
    print(adjacency_matrix)
    print("\nEdge index:")
    print(edge_index)

    print("\n=== TOKENS ===")
    print(tokens)
    print("\nToken text:")
    print(tokens_to_text(tokens, tokenizer))

    print("\n=== RECONSTRUCTION FROM TOKENS ===")
    print("Reconstructed edge index:")
    print(reconstructed_edge_index)
    print("\nReconstructed adjacency matrix:")
    print(reconstructed_adjacency)


if __name__ == "__main__":
    main()
