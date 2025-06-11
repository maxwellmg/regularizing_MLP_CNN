import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Internal Imports
from metrics.embedding_metrics import compute_embedding_similarity
from regularization.distances import *

distance_list = ['baseline', 'euclidean', 'manhattan', 'cosine', 'minkowski', 'chebyshev', 'canberra', 'bray-curtis', 'hamming', 'mahalanobis']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


def train(model, loader, optimizer, distance_type, criterion, epoch, lamb):
    model.train()
    total_loss_epoch = 0
    total_correct = 0
    total_samples = 0
    total_reg = 0
    total_emb_norm = 0
    batch_count = 0
    total_similarity = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output, embedding = model(data, return_embedding=True)
        loss = criterion(output, target)

        if distance_type in distance_list:

            # Embedding Similarity
            sim_score = compute_embedding_similarity(embedding)
            total_similarity += sim_score

            # Distance Measurements
            if distance_type == 'euclidean':
                embd_reg_unique_distance = euclidean_distance(embedding)
            elif distance_type == 'manhattan':
                embd_reg_unique_distance = manhattan_distance(embedding)
            elif distance_type == 'cosine':
                embd_reg_unique_distance = cosine_distance(embedding)
            elif distance_type == 'minkowski':
                embd_reg_unique_distance = minkowski_distance(embedding)
            elif distance_type == 'chebyshev':
                embd_reg_unique_distance = chebyshev_distance(embedding)
            elif distance_type == 'canberra':
                embd_reg_unique_distance = canberra_distance(embedding)
            elif distance_type == 'bray-curtis':
                embd_reg_unique_distance = bray_curtis_distance(embedding)
            elif distance_type == 'hamming':
                embd_reg_unique_distance = hamming_distance(embedding)
            elif distance_type == 'mahalanobis':
                embd_reg_unique_distance = mahalanobis_distance(embedding)
            elif distance_type == 'baseline':
                embd_reg_unique_distance = no_regularization(embedding)

            reg = embd_reg_unique_distance
            total_reg += reg.item()
            total_emb_norm += torch.mean(torch.norm(embedding, dim=1)).item()
            total_loss = loss + lamb * reg # Apply regularization

        else: # If no embedding is returned, only use the classification loss
            total_loss = loss


        total_loss.backward()
        optimizer.step()

        total_loss_epoch += total_loss.item()

        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        total_samples += target.size(0)
        batch_count += 1

        '''if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}] "
                  f"Loss: {loss.item():.4f} | Reg: {reg.item():.4f} | Total: {total_loss.item():.4f}")'''

    train_acc = 100.0 * total_correct / total_samples
    avg_loss = total_loss_epoch / batch_count
    avg_reg = total_reg / batch_count
    avg_emb_norm = total_emb_norm / batch_count
    avg_similarity = total_similarity / batch_count

    return {
        'epoch': epoch,
        'train_loss': round(avg_loss, 5),
        'train_acc': round(train_acc, 5),
        'reg_term': round(avg_reg, 5),
        'embedding_norm': round(avg_emb_norm, 5),
        'embedding_similarity': round(avg_similarity, 5)
        }

