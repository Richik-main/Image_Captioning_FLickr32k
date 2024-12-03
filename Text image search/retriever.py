import torch
from torch import nn
from preprocess import text_tokenizer, text_encoder,data
import torch.nn.functional as F
import re
from matplotlib import pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")
image_embeddings_dict = torch.load("image_embeddings.pt")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 768  # Both ViT and BERT output 768-dimensional embeddings
projection_dim = 512  # Dimension of the shared embedding space


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super(ProjectionHead, self).__init__()
        self.iniprojection = nn.Linear(embedding_dim, 1024)
        self.relu = nn.ReLU()
        self.projection = nn.Linear(1024, projection_dim)

    def forward(self, x):
        x = self.iniprojection(x)    # Linear transformation
        x = self.relu(x)              # ReLU activation
        x = self.projection(x)       # Second linear transformation
        x = F.normalize(x, p=2, dim=-1)  # L2 Normalization
        return x

# Instantiate projection heads
image_projection = ProjectionHead(embedding_dim, projection_dim).to(device)
text_projection = ProjectionHead(embedding_dim, projection_dim).to(device)
#
#
#

# Specify the correct path to your checkpoint file
checkpoint_path = 'best_model_checkpoint.pth'  # Ensure this is the correct path

# Check if the checkpoint file exists
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'. Please verify the path.")

# Load the checkpoint
try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("Checkpoint loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")
except Exception as e:
    raise RuntimeError(f"Error loading checkpoint: {e}")

# Inspect checkpoint keys (optional, for debugging)
print("Checkpoint keys:", checkpoint.keys())

# Load state dictionaries for projection heads
# Use 'image_projection_state_dict' and 'text_projection_state_dict' keys
if 'image_projection_state_dict' in checkpoint and 'text_projection_state_dict' in checkpoint:
    image_projection.load_state_dict(checkpoint['image_projection_state_dict'])
    text_projection.load_state_dict(checkpoint['text_projection_state_dict'])
    print("Projection heads loaded successfully.")
else:
    raise KeyError("Checkpoint does not contain 'image_projection_state_dict' and 'text_projection_state_dict' keys.")

# If your text_encoder was fine-tuned and saved in the checkpoint, load it
# Assuming you have not saved 'text_encoder_state_dict', we'll skip this step
# If you have, uncomment and ensure it's saved correctly during checkpointing

# if 'text_encoder_state_dict' in checkpoint:
#     text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
#     text_encoder.to(device)
#     text_encoder.eval()
#     print("Text encoder loaded successfully.")
# else:
#     # If using a pre-trained text_encoder without fine-tuning
#     text_encoder.to(device)
#     text_encoder.eval()
#     print("Using pre-trained text encoder without fine-tuning.")

# Move projection heads to the appropriate device and set to evaluation mode
image_projection.to(device)
text_projection.to(device)
image_projection.eval()
text_projection.eval()

#
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#
#     # Load state dictionaries
#     image_projection.load_state_dict(checkpoint['image_projection_state_dict'])
#     text_projection.load_state_dict(checkpoint['text_projection_state_dict'])
#
#     # If you have an optimizer and want to load its state
#     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#     # Retrieve epoch and best_loss if needed
#     start_epoch = checkpoint.get('epoch', 0) + 1
#     best_loss = checkpoint.get('best_loss', float('inf'))
#
#     print(f"Loaded checkpoint '{checkpoint_path}' (Epoch {checkpoint.get('epoch', 0)}) with loss {best_loss:.4f}")
# else:
#     start_epoch = 0
#     best_loss = float('inf')
#     print("No checkpoint found. Starting with randomly initialized projection heads.")
def encode_texts(input_ids, attention_masks):
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    text_projection.eval()
    with torch.no_grad():
        text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_masks)
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        text_proj_embeddings = text_projection(text_embeddings)
    return text_proj_embeddings.cpu()









# ------------------------
# Text-to-Image Retrieval
# ------------------------
def preprocess_text(caption, tokenizer, max_length=32):
    # Basic cleaning
    caption = caption.lower()
    caption = re.sub(r'[^a-zA-Z0-9\s,.!?\'"-]', '', caption)
    caption = re.sub(r'\s+', ' ', caption).strip()

    # Tokenize and encode
    tokens = tokenizer(
        caption,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = tokens['input_ids'].squeeze()
    attention_mask = tokens['attention_mask'].squeeze()
    return input_ids, attention_mask

def retrieve_images(query_caption, top_k=5):
    # Encode query caption
    input_id, attention_mask = preprocess_text(query_caption, tokenizer=text_tokenizer, max_length=32)
    input_id = input_id.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    query_embedding = encode_texts(input_id, attention_mask)

    # Compute similarities
    all_image_embeddings = torch.stack(list(image_embeddings_dict.values()))
    similarities = torch.matmul(query_embedding, all_image_embeddings.T).squeeze(0)
    similarities = similarities.cpu()

    # Get top_k results
    top_k_indices = similarities.topk(top_k).indices
    retrieved_img_ids = [list(image_embeddings_dict.keys())[idx] for idx in top_k_indices]
    return retrieved_img_ids
def fetch_and_plot_images(retrieved_img_ids, dataset):
    """
    Fetch images from the dataset based on the retrieved `img_id`s and plot them.

    Args:
        retrieved_img_ids (list): List of `img_id`s retrieved by `retrieve_images`.
        dataset: The dataset containing image data.

    Returns:
        None
    """
    # Fetch images corresponding to the retrieved img_ids
    retrieved_images = []
    for img_id in retrieved_img_ids:
        # Find the matching entry in the dataset
        img_data = next((data for data in dataset if data['img_id'] == img_id), None)
        if img_data:
            retrieved_images.append((img_id, img_data['image']))

    if not retrieved_images:
        print("No images found for the retrieved img_ids.")
        return

    # Plot the images
    num_images = len(retrieved_images)
    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    if num_images == 1:
        axs = [axs]  # Ensure axs is iterable for a single image

    for idx, (img_id, pil_img) in enumerate(retrieved_images):  # Image is already in PIL format
        axs[idx].imshow(pil_img)
        axs[idx].axis('off')
        axs[idx].set_title(f"Image ID: {img_id}")

    plt.tight_layout()
    plt.show()


# Example usage
query = "two dogs fighting in the snow and being cute"
retrieved_images = retrieve_images(query, top_k=5)

print(f"Retrieved Image IDs for query '{query}': {retrieved_images}")
fetch_and_plot_images(retrieved_images, data)
