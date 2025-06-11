import torch

def compute_embedding_similarity(embedding):
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    sim_matrix = torch.matmul(embedding, embedding.T)
    B = sim_matrix.size(0)
    # Remove diagonal (self-similarity)
    sim_matrix = sim_matrix - torch.eye(B, device=embedding.device)
    return sim_matrix.abs().mean().item()

