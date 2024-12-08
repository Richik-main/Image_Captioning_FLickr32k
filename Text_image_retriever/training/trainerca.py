
import torch
from tqdm import tqdm
from models.contrastive_loss import contrastive_loss
from utils.utils import save_checkpoint

def orthogonal_regularization(embeddings1, embeddings2):
    """Compute Orthogonal Regularization Loss."""
    batch_size = embeddings1.size(0)
    orth_loss = 0.0
    for i in range(batch_size):
        for j in range(batch_size):
            if i != j:  # Negative pairs
                dot_product = torch.dot(embeddings1[i], embeddings2[j])
                orth_loss += dot_product ** 2
    orth_loss /= (batch_size * (batch_size - 1))
    return orth_loss

def center_loss(embeddings1, embeddings2):
    """Compute Center Loss."""
    batch_size = embeddings1.size(0)
    center_loss = 0.0
    for i in range(batch_size):
        center = (embeddings1[i] + embeddings2[i]) / 2
        center_loss += (
            torch.norm(embeddings1[i] - center) ** 2 +
            torch.norm(embeddings2[i] - center) ** 2
        ) / 2
    center_loss /= batch_size
    return center_loss

#
# def train(model_dict, data_loader, optimizer, device, num_epochs, checkpoint_path, start_epoch=0, best_loss=float('inf'), best_model_states=None, lambda_orth=0.01, lambda_center=0.01):
#     if best_model_states is None:
#         best_model_states = {k: v.state_dict() for k, v in model_dict.items()}
#
#     for epoch in range(start_epoch, num_epochs):
#         total_loss = 0.0
#         progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
#
#         for batch in progress_bar:
#             img_ids, images, input_ids, attention_masks = batch
#             images = images.to(device)
#             input_ids = input_ids.to(device)
#             attention_masks = attention_masks.to(device)
#
#             optimizer.zero_grad()
#
#             # Forward pass
#             image_outputs = model_dict['image_encoder'](images)
#             image_embeddings = image_outputs.last_hidden_state[:, 0]  # CLS token
#
#             text_outputs = model_dict['text_encoder'](input_ids=input_ids, attention_mask=attention_masks)
#             text_embeddings = text_outputs.last_hidden_state[:, 0]  # CLS token
#
#             image_proj = model_dict['image_projection'](image_embeddings)
#             text_proj = model_dict['text_projection'](text_embeddings)
#
#             # Loss
#             loss = contrastive_loss(image_proj, text_proj)
#             # orth_loss = orthogonal_regularization(image_embeddings, text_embeddings)
#             # center_loss_value = center_loss(image_embeddings, text_embeddings)
#             # total_loss_reg = (loss + lambda_orth * orth_loss + lambda_center * center_loss_value)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
#
#         avg_loss = total_loss / len(data_loader)
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             best_model_states = {k: v.state_dict() for k, v in model_dict.items()}
#             save_checkpoint({
#                 'epoch': epoch,
#                 'best_loss': best_loss,
#                 'model_states': best_model_states,
#                 'optimizer_state_dict': optimizer.state_dict()
#             }, checkpoint_path)
#
#         print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
#
#     # Return the best model states after training
#     return best_model_states

def train(model_dict, data_loader, optimizer, device, num_epochs, checkpoint_path, start_epoch=0, best_loss=float('inf'), best_model_states=None, lambda_orth=0.01, lambda_center=0.01):
    if best_model_states is None:
        best_model_states = {k: v.state_dict() for k, v in model_dict.items()}

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            img_ids, images, input_ids, attention_masks = batch
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            image_outputs = model_dict['image_encoder'](images)
            image_embeddings = image_outputs.last_hidden_state[:, 0]  # CLS token

            text_outputs = model_dict['text_encoder'](input_ids=input_ids, attention_mask=attention_masks)
            text_embeddings = text_outputs.last_hidden_state[:, 0]  # CLS token

            image_proj = model_dict['image_projection'](image_embeddings)
            text_proj = model_dict['text_projection'](text_embeddings)

            # Apply cross-attention: images attend to text
            image_attends_text = model_dict['cross_attention'](query=image_proj.unsqueeze(1),
                                                               key=text_proj.unsqueeze(1),
                                                               value=text_proj.unsqueeze(1))
            image_attends_text = image_attends_text.squeeze(1)  # Remove the extra sequence dimension

            # Apply cross-attention: text attends to images
            text_attends_image = model_dict['cross_attention'](query=text_proj.unsqueeze(1),
                                                               key=image_proj.unsqueeze(1),
                                                               value=image_proj.unsqueeze(1))
            text_attends_image = text_attends_image.squeeze(1)  # Remove the extra sequence dimension

            # Compute contrastive loss
            loss = contrastive_loss(image_attends_text, text_attends_image)


            orth_loss = orthogonal_regularization(image_embeddings, text_embeddings)
            center_loss_value = center_loss(image_embeddings, text_embeddings)
            total_loss_reg = (loss + lambda_orth * orth_loss + lambda_center * center_loss_value)
            total_loss_reg.backward()
            optimizer.step()

            total_loss += total_loss_reg.item()
            progress_bar.set_postfix({'Loss': f"{total_loss_reg.item():.4f}"})

        avg_loss = total_loss / len(data_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_states = {k: v.state_dict() for k, v in model_dict.items()}
            save_checkpoint({
                'epoch': epoch,
                'best_loss': best_loss,
                'model_states': best_model_states,
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    # Return the best model states after training
    return best_model_states
