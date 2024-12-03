# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from preprocess import data_loader, text_encoder, image_encoder
# from tqdm import tqdm
#
# # ------------------------
# # Device Configuration
# # ------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # ------------------------
# # Projection Layers
# # ------------------------
# embedding_dim = 768  # ViT and BERT output 768-dimensional embeddings
# projection_dim = 512  # Dimension of the shared embedding space
#
# class ProjectionHead(nn.Module):
#     def __init__(self, embedding_dim, projection_dim):
#         super(ProjectionHead, self).__init__()
#         self.iniprojection = nn.Linear(embedding_dim, 1024)
#         self.relu = nn.ReLU()
#         self.projection = nn.Linear(1024, projection_dim)
#
#     def forward(self, x):
#         x = self.iniprojection(x)    # Linear transformation
#         x = self.relu(x)              # ReLU activation
#         x = self.projection(x)       # Second linear transformation
#         x = F.normalize(x, p=2, dim=-1)  # L2 Normalization
#         return x
#
#
# # Instantiate projection heads
# image_projection = ProjectionHead(embedding_dim, projection_dim).to(device)
# text_projection = ProjectionHead(embedding_dim, projection_dim).to(device)
#
# print("Weights before training:")
# print("Image Projection:", image_projection.projection.weight)
# print("Text Projection:", text_projection.projection.weight)
#
# # ------------------------
# # Optimizer and Loss
# # ------------------------
# optimizer = torch.optim.AdamW(
#     list(image_projection.parameters()) + list(text_projection.parameters()),
#     lr=8e-5
# )
#
# temperature = 0.07  # Hyperparameter for scaling similarities
#
# def contrastive_loss(image_embeddings, text_embeddings):
#     # Normalize embeddings to ensure consistency
#     image_embeddings = F.normalize(image_embeddings, dim=1)
#     text_embeddings = F.normalize(text_embeddings, dim=1)
#
#     # Compute similarity logits
#     logits = torch.matmul(text_embeddings, image_embeddings.T) / temperature
#     batch_size = image_embeddings.size(0)
#
#     # Labels are indices of matching pairs
#     labels = torch.arange(batch_size, device=device)
#
#     # Cross-entropy loss for both directions
#     loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
#     return loss
# # ------------
# # Run the model from last best saved path
# # ------------
#
# # Function to load checkpoint
# def load_checkpoint(checkpoint_path, image_model, text_model, optimizer):
#     """
#     Loads the checkpoint from the specified path.
#
#     Args:
#         checkpoint_path (str): Path to the checkpoint file.
#         image_model (nn.Module): Image projection model.
#         text_model (nn.Module): Text projection model.
#         optimizer (Optimizer): Optimizer.
#
#     Returns:
#         int: The epoch to start from.
#         float: The best loss so far.
#     """
#     if torch.cuda.is_available():
#         checkpoint = torch.load(checkpoint_path)
#     else:
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#
#     image_model.load_state_dict(checkpoint['image_projection_state_dict'])
#     text_model.load_state_dict(checkpoint['text_projection_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch'] + 1
#     best_loss = checkpoint['loss']
#     print(f"Loaded checkpoint '{checkpoint_path}' (Epoch {checkpoint['epoch']})")
#     return start_epoch, best_loss
#
#
# # Path to your checkpoint
# checkpoint_path = 'best_model_checkpoint.pth'
#
# # Check if checkpoint exists and load it
# import os
#
# if os.path.exists(checkpoint_path):
#     start_epoch, best_loss = load_checkpoint(checkpoint_path, image_projection, text_projection, optimizer)
# else:
#     start_epoch = 0
#     best_loss = float('inf')
#     print("No checkpoint found. Starting training from scratch.")
#
#
# # ------------------------
# # Save Best Model Function
# # ------------------------
#
# # Initialize variables to track the best model
# best_loss = float('inf')
# best_model_path = "best_projection_model.pth"
#
# # ------------------------
# # Training Loop
# # ------------------------
# num_epochs = 20
# for epoch in range(num_epochs):
#     total_loss = 0.0
#
#     # Iterate over data loader
#     for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
#         img_ids, images, input_ids, attention_masks = batch
#         images = images.to(device)
#         input_ids = input_ids.to(device)
#         attention_masks = attention_masks.to(device)
#
#         # Encode images
#         with torch.no_grad():
#             image_embeddings = image_encoder(pixel_values=images).last_hidden_state[:, 0, :]
#
#         # Encode text
#         with torch.no_grad():
#             text_embeddings = text_encoder(input_ids=input_ids, attention_mask=attention_masks).last_hidden_state[:, 0, :]
#
#         # Project embeddings to shared space
#         image_proj_embeddings = image_projection(image_embeddings)
#         text_proj_embeddings = text_projection(text_embeddings)
#
#         # Compute loss
#         loss = contrastive_loss(image_proj_embeddings, text_proj_embeddings)
#         total_loss += loss.item()
#
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     avg_loss = total_loss / len(data_loader)
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         torch.save({
#             "image_projection": image_projection.state_dict(),
#             "text_projection": text_projection.state_dict(),
#             "optimizer": optimizer.state_dict()
#         }, best_model_path)
#         print(f"New best model saved with loss: {avg_loss:.4f}")
#
#
# # ------------------------
# # Save Embeddings
# # ------------------------
#
#
# def encode_data(data_loader, encoder, projection, encode_type="images"):
#     """
#     Encodes all data in a data loader using the specified encoder and projection layer.
#
#     Args:
#         data_loader: DataLoader containing images or text data.
#         encoder: The pre-trained encoder (image_encoder or text_encoder).
#         projection: The projection head corresponding to the encoder.
#         encode_type: "images" or "texts" to specify data type.
#
#     Returns:
#         A dictionary of embeddings.
#     """
#     embeddings_dict = {}
#     for batch in tqdm(data_loader, desc=f"Encoding {encode_type.capitalize()}"):
#         img_ids, images, input_ids, attention_masks = batch
#
#         # Select inputs based on data type
#         inputs = images if encode_type == "images" else (input_ids, attention_masks)
#         inputs = [item.to(device) for item in inputs] if encode_type == "texts" else inputs.to(device)
#
#         # Encode and project embeddings
#         with torch.no_grad():
#             if encode_type == "images":
#                 embeddings = encoder(pixel_values=inputs).last_hidden_state[:, 0, :]
#             else:
#                 input_ids, attention_masks = inputs
#                 embeddings = encoder(input_ids=input_ids, attention_mask=attention_masks).last_hidden_state[:, 0, :]
#
#             projected_embeddings = projection(embeddings).cpu()
#
#         # Save embeddings in the dictionary
#         for img_id, embedding in zip(img_ids, projected_embeddings):
#             if encode_type == "texts":
#                 if img_id not in embeddings_dict:
#                     embeddings_dict[img_id] = []
#                 embeddings_dict[img_id].append(embedding)
#             else:
#                 embeddings_dict[img_id] = embedding
#
#     return embeddings_dict
# # # Later, to load the best model
# # checkpoint = torch.load(best_model_path)
# # image_projection.load_state_dict(checkpoint["image_projection"])
# # text_projection.load_state_dict(checkpoint["text_projection"])
# # optimizer.load_state_dict(checkpoint["optimizer"])
# # image_projection.to(device)
# # text_projection.to(device)
# # Save image and text embeddings
# image_embeddings_dict = encode_data(data_loader, image_encoder, image_projection, encode_type="images")
# torch.save(image_embeddings_dict, 'image_embeddings.pt')
#
# text_embeddings_dict = encode_data(data_loader, text_encoder, text_projection, encode_type="texts")
# torch.save(text_embeddings_dict, 'text_embeddings.pt')
#
# print("Saved embeddings successfully.")


# ------------------------
# Projection Layers
# ------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import data_loader, text_encoder, image_encoder  # Ensure these are properly defined and loaded
from tqdm import tqdm
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import math

# ------------------------
# Device Configuration
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------
# Projection Layers
# ------------------------
embedding_dim = 768  # ViT and BERT output 768-dimensional embeddings
projection_dim = 512  # Dimension of the shared embedding space


def lecun_normal_(tensor):
    """
    Initializes the tensor using LeCun Normal Initialization, suitable for SELU activations.

    Args:
        tensor (torch.Tensor): The tensor to initialize.

    Returns:
        torch.Tensor: The initialized tensor.
    """
    fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
    std = math.sqrt(1.0 / fan_in)
    with torch.no_grad():
        tensor.normal_(0, std)
    return tensor


#
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super(ProjectionHead, self).__init__()
        self.iniprojection = nn.Linear(embedding_dim, 1024)
        self.relu = nn.ReLU()
        self.projection = nn.Linear(1024, projection_dim)
        #self.init_weights()
#     def init_weights(self):
#         """
#         Initializes weights using LeCun Normal Initialization for SELU activations
#         and zeros for biases.
#         """
#         lecun_normal_(self.iniprojection.weight)
#         lecun_normal_(self.projection.weight)
#         # Initialize biases to zero
#         if self.iniprojection.bias is not None:
#             nn.init.zeros_(self.iniprojection.bias)
#         if self.projection.bias is not None:
#             nn.init.zeros_(self.projection.bias)
#
    def forward(self, x):
        x = self.iniprojection(x)    # Linear transformation
        x = self.relu(x)             # ReLU activation
        x = self.projection(x)       # Second linear transformation
        x = F.normalize(x, p=2, dim=-1)  # L2 Normalization
        return x
# class ProjectionHead(nn.Module):
#     def __init__(self, embedding_dim, projection_dim):
#         super(ProjectionHead, self).__init__()
#         self.iniprojection = nn.Linear(embedding_dim, 1024)
#         self.selu = nn.SELU()
#         self.projection = nn.Linear(1024, projection_dim)
#         self.init_weights()
#
#     def init_weights(self):
#         """
#         Initializes weights using LeCun Normal Initialization for SELU activations
#         and zeros for biases.
#         """
#         lecun_normal_(self.iniprojection.weight)
#         lecun_normal_(self.projection.weight)
#         # Initialize biases to zero
#         if self.iniprojection.bias is not None:
#             nn.init.zeros_(self.iniprojection.bias)
#         if self.projection.bias is not None:
#             nn.init.zeros_(self.projection.bias)
#
#     def forward(self, x):
#         x = self.iniprojection(x)
#         x = self.selu(x)
#         x = self.projection(x)
#         x = F.normalize(x, p=2, dim=-1)  # L2 Normalization
#         return x


# Instantiate projection heads
image_projection = ProjectionHead(embedding_dim, projection_dim).to(device)
text_projection = ProjectionHead(embedding_dim, projection_dim).to(device)

# Verify weight initialization
print("\nWeights after initialization:")
print("Image Projection Weights Mean:", image_projection.projection.weight.mean().item())
print("Image Projection Weights Std Dev:", image_projection.projection.weight.std().item())
print("Text Projection Weights Mean:", text_projection.projection.weight.mean().item())
print("Text Projection Weights Std Dev:", text_projection.projection.weight.std().item())

# ------------------------
# Optimizer and Loss
# ------------------------
optimizer = torch.optim.AdamW(
    list(image_projection.parameters()) + list(text_projection.parameters()),
    lr=1e-4
)

temperature = 0.07  # Hyperparameter for scaling similarities


def contrastive_loss(image_embeddings, text_embeddings):
    """
    Computes the contrastive loss between image and text embeddings.

    Args:
        image_embeddings (torch.Tensor): Projected image embeddings.
        text_embeddings (torch.Tensor): Projected text embeddings.

    Returns:
        torch.Tensor: The computed contrastive loss.
    """
    # Normalize embeddings to ensure consistency
    image_embeddings = F.normalize(image_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)

    # Compute similarity logits
    logits = torch.matmul(text_embeddings, image_embeddings.T) / temperature
    batch_size = image_embeddings.size(0)

    # Labels are indices of matching pairs
    labels = torch.arange(batch_size, device=device)

    # Cross-entropy loss for both directions
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss


# ------------------------
# Checkpoint Functions
# ------------------------

def save_checkpoint(state, filename='checkpoint.pth'):
    """
    Saves the training state to a file.

    Args:
        state (dict): State dictionary containing model states, optimizer state, epoch, etc.
        filename (str): Filename to save the checkpoint.
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, image_model, text_model, optimizer):
    """
    Loads the checkpoint from the specified path.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        image_model (nn.Module): Image projection model.
        text_model (nn.Module): Text projection model.
        optimizer (Optimizer): Optimizer.

    Returns:
        int: The epoch to start from.
        float: The best loss so far.
    """
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    image_model.load_state_dict(checkpoint['image_projection_state_dict'])
    text_model.load_state_dict(checkpoint['text_projection_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    print(f"Loaded checkpoint '{checkpoint_path}' (Epoch {checkpoint['epoch']}) with loss {best_loss:.4f}")
    return start_epoch, best_loss


# Path to your checkpoint
checkpoint_path = 'best_model_checkpoint.pth'

# Initialize variables to track the best model
best_loss = float('inf')  # Initialize to infinity

# Initialize TensorBoard writer
writer = SummaryWriter('runs/contrastive_training_selu')

# Check if checkpoint exists and load it
if os.path.exists(checkpoint_path):
    start_epoch, loaded_best_loss = load_checkpoint(checkpoint_path, image_projection, text_projection, optimizer)
    best_loss = loaded_best_loss  # Update best_loss from checkpoint
else:
    start_epoch = 0
    print("No checkpoint found. Starting training from scratch.")

# ------------------------
# Training Loop
# ------------------------
num_epochs = 30
total_epochs = num_epochs  # Total number of epochs you want to train

for epoch in range(start_epoch, total_epochs):
    total_loss = 0.0
    image_projection.train()
    text_projection.train()
    batch_losses = []

    # Initialize tqdm progress bar for the epoch
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        img_ids, images, input_ids, attention_masks = batch
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        # Encode images
        with torch.no_grad():
            image_outputs = image_encoder(pixel_values=images)
            # Adjust based on your image_encoder's output structure
            if hasattr(image_outputs, 'last_hidden_state'):
                image_embeddings = image_outputs.last_hidden_state[:, 0, :]
            elif hasattr(image_outputs, 'pooler_output'):
                image_embeddings = image_outputs.pooler_output
            else:
                raise AttributeError("Image encoder output does not have 'last_hidden_state' or 'pooler_output'.")

        # Encode text
        with torch.no_grad():
            text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_masks)
            # Adjust based on your text_encoder's output structure
            if hasattr(text_outputs, 'last_hidden_state'):
                text_embeddings = text_outputs.last_hidden_state[:, 0, :]
            elif hasattr(text_outputs, 'pooler_output'):
                text_embeddings = text_outputs.pooler_output
            else:
                raise AttributeError("Text encoder output does not have 'last_hidden_state' or 'pooler_output'.")

        # Project embeddings to shared space
        image_proj_embeddings = image_projection(image_embeddings)
        text_proj_embeddings = text_projection(text_embeddings)

        # Compute loss
        loss = contrastive_loss(image_proj_embeddings, text_proj_embeddings)
        total_loss += loss.item()
        batch_losses.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar with the current batch loss
        progress_bar.set_postfix({'Batch Loss': f"{loss.item():.4f}"})

        # Log batch loss to TensorBoard
        global_step = epoch * len(data_loader) + batch_idx
        writer.add_scalar('Batch Loss', loss.item(), global_step)

    # Calculate average loss for the epoch
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch + 1}/{total_epochs}, Average Loss: {avg_loss:.4f}")

    # Compute batch-wise loss statistics
    batch_mean_loss = np.mean(batch_losses)
    batch_std_loss = np.std(batch_losses)
    print(f"Batch Loss - Mean: {batch_mean_loss:.4f}, Std Dev: {batch_std_loss:.4f}")

    # Log average epoch loss and statistics to TensorBoard
    writer.add_scalar('Average Epoch Loss', avg_loss, epoch)
    writer.add_scalar('Batch Loss Mean', batch_mean_loss, epoch)
    writer.add_scalar('Batch Loss Std Dev', batch_std_loss, epoch)

    # Check if current loss is the best
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_checkpoint({
            'epoch': epoch,
            'best_loss': best_loss,
            'image_projection_state_dict': image_projection.state_dict(),
            'text_projection_state_dict': text_projection.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, filename=checkpoint_path)
        print(f"New best model saved with loss: {avg_loss:.4f}")
        # Log best loss to TensorBoard
        writer.add_scalar('Best Loss', best_loss, epoch)

    # Optionally, save intermediate checkpoints (e.g., every 5 epochs)
    # if (epoch + 1) % 5 == 0:
    #     intermediate_checkpoint = f'checkpoint_epoch_{epoch+1}.pth'
    #     save_checkpoint({
    #         'epoch': epoch,
    #         'best_loss': best_loss,
    #         'image_projection_state_dict': image_projection.state_dict(),
    #         'text_projection_state_dict': text_projection.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict()
    #     }, filename=intermediate_checkpoint)
    #     print(f"Intermediate checkpoint saved at epoch {epoch + 1}")

print("Training completed.")
writer.close()


# ------------------------
# Save Embeddings
# ------------------------

def encode_data(data_loader, encoder, projection, encode_type="images"):
    """
    Encodes all data in a data loader using the specified encoder and projection layer.

    Args:
        data_loader (DataLoader): DataLoader containing images or text data.
        encoder (nn.Module): The pre-trained encoder (image_encoder or text_encoder).
        projection (nn.Module): The projection head corresponding to the encoder.
        encode_type (str): "images" or "texts" to specify data type.

    Returns:
        dict: A dictionary mapping IDs to their corresponding embeddings.
    """
    embeddings_dict = {}
    projection.eval()  # Set projection head to evaluation mode
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Encoding {encode_type.capitalize()}"):
            img_ids, images, input_ids, attention_masks = batch

            if encode_type == "images":
                images = images.to(device)
                image_outputs = encoder(pixel_values=images)
                # Adjust based on your image_encoder's output structure
                if hasattr(image_outputs, 'last_hidden_state'):
                    embeddings = image_outputs.last_hidden_state[:, 0, :]
                elif hasattr(image_outputs, 'pooler_output'):
                    embeddings = image_outputs.pooler_output
                else:
                    raise AttributeError("Image encoder output does not have 'last_hidden_state' or 'pooler_output'.")

                projected_embeddings = projection(embeddings).cpu()

                # Save embeddings in the dictionary
                for img_id, embedding in zip(img_ids, projected_embeddings):
                    embeddings_dict[img_id] = embedding

            elif encode_type == "texts":
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                text_outputs = encoder(input_ids=input_ids, attention_mask=attention_masks)
                # Adjust based on your text_encoder's output structure
                if hasattr(text_outputs, 'last_hidden_state'):
                    embeddings = text_outputs.last_hidden_state[:, 0, :]
                elif hasattr(text_outputs, 'pooler_output'):
                    embeddings = text_outputs.pooler_output
                else:
                    raise AttributeError("Text encoder output does not have 'last_hidden_state' or 'pooler_output'.")

                projected_embeddings = projection(embeddings).cpu()

                # Save embeddings in the dictionary
                for img_id, embedding in zip(img_ids, projected_embeddings):
                    if img_id not in embeddings_dict:
                        embeddings_dict[img_id] = []
                    embeddings_dict[img_id].append(embedding)
            else:
                raise ValueError("encode_type must be either 'images' or 'texts'.")

    return embeddings_dict


# Save image and text embeddings
print("\nEncoding and saving image embeddings...")
image_embeddings_dict = encode_data(data_loader, image_encoder, image_projection, encode_type="images")
torch.save(image_embeddings_dict, 'image_embeddings.pt')

print("Encoding and saving text embeddings...")
text_embeddings_dict = encode_data(data_loader, text_encoder, text_projection, encode_type="texts")
torch.save(text_embeddings_dict, 'text_embeddings.pt')

print("Saved embeddings successfully.")
