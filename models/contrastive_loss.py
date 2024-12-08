import torch.nn.functional as F
import torch

def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    image_embeddings = F.normalize(image_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)

    logits = torch.matmul(text_embeddings, image_embeddings.T) / temperature
    labels = torch.arange(image_embeddings.size(0), device=image_embeddings.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss
