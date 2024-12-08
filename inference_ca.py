import torch
import os
import re
from transformers import AutoTokenizer, AutoModel, ViTModel
from datasets import load_from_disk
from models.projection_head import ProjectionHead
from models.Crossattention import CrossAttentionModule
from torch.nn.functional import normalize
from matplotlib import pyplot as plt
from tqdm import tqdm
import h5py

# ------------------------
# Configuration
# ------------------------

EMBEDDING_DIM = 768
PROJECTION_DIM = 512
NUM_HEADS = 8
CHECKPOINT_PATH = 'best_model_checkpoint_modular_CroA.pth'
EMBEDDINGS_PATH = 'best_embeddings_CroA.pt'
DATA_PATH = "/home/ubuntu/Training_img_cap/Caption_module/flickr30k_dataset_backup"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ------------------------
# Loading Functions
# ------------------------

def load_embeddings_hdf5(embeddings_path):
    with h5py.File(embeddings_path, 'r') as h5_file:
        img_ids = h5_file['img_ids'][:]
        img_ids = [id.decode('utf8') for id in img_ids]
        image_embeddings = torch.tensor(h5_file['image_embeddings'][:])
    return img_ids, normalize(image_embeddings, p=2, dim=-1)


# ------------------------
# Model Initialization
# ------------------------

def initialize_models(device):
    print("Loading models...")
    image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
    text_encoder = AutoModel.from_pretrained('bert-base-uncased').to(device)
    image_projection = ProjectionHead(EMBEDDING_DIM, PROJECTION_DIM).to(device)
    text_projection = ProjectionHead(EMBEDDING_DIM, PROJECTION_DIM).to(device)
    cross_attention = CrossAttentionModule(embed_dim=PROJECTION_DIM, num_heads=NUM_HEADS).to(device)

    return image_encoder, text_encoder, image_projection, text_projection, cross_attention


# ------------------------
# Text Preprocessing Functions
# ------------------------

def preprocess_text(caption, tokenizer, max_length=32):
    caption = caption.lower()
    caption = re.sub(r'[^a-zA-Z0-9\s,.!?\'"-]', '', caption)
    caption = re.sub(r'\s+', ' ', caption).strip()

    tokens = tokenizer(
        caption,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()

def encode_texts(input_ids, attention_masks, text_encoder, text_projection, device):
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)
    with torch.no_grad():
        text_outputs = text_encoder(input_ids=input_ids.unsqueeze(0), attention_mask=attention_masks.unsqueeze(0))
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        text_proj_embeddings = text_projection(text_embeddings)
    return text_proj_embeddings


# ------------------------
# Retrieval Functions
# ------------------------

def retrieve_images(query_caption, tokenizer, text_encoder, text_projection, image_embeddings, img_ids, device, top_k):
    # Encode query caption
    input_ids, attention_mask = preprocess_text(query_caption, tokenizer)
    query_embedding = encode_texts(input_ids, attention_mask, text_encoder, text_projection, device)
    query_embedding = normalize(query_embedding, p=2, dim=-1)

    # Compute similarities
    image_embeddings = image_embeddings.to(device)
    similarities = torch.matmul(query_embedding, image_embeddings.T).squeeze(0)
    similarities = similarities.cpu()

    # Sort indices by similarities
    sorted_indices = similarities.argsort(descending=True)
    retrieved_img_ids = []
    seen_img_ids = set()
    for idx in sorted_indices:
        img_id = img_ids[idx]
        if img_id not in seen_img_ids:
            retrieved_img_ids.append(img_id)
            seen_img_ids.add(img_id)
            if len(retrieved_img_ids) == top_k:
                break

    return retrieved_img_ids

def fetch_and_plot_images(retrieved_img_ids, dataset):
    retrieved_images = []
    for img_id in retrieved_img_ids:
        img_data = next((data for data in dataset if data['img_id'] == img_id), None)
        if img_data:
            retrieved_images.append((img_id, img_data['image']))

    if not retrieved_images:
        print("No images found for the retrieved img_ids.")
        return

    num_images = len(retrieved_images)
    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    if num_images == 1:
        axs = [axs]  # Ensure axs is iterable for a single image

    for idx, (img_id, pil_img) in enumerate(retrieved_images):
        axs[idx].imshow(pil_img)
        axs[idx].axis('off')
        axs[idx].set_title(f"Image ID: {img_id}")

    plt.tight_layout()
    plt.show()


# ------------------------
# Recall@k Calculation
# ------------------------

def calculate_recall_at_k(queries, ground_truth, tokenizer, text_encoder, text_projection, cross_attention,
                          image_embeddings, img_ids, device, k):
    correct_retrievals = 0

    for query_caption in tqdm(queries, desc="Evaluating Recall@k"):
        retrieved_img_ids = retrieve_images(
            query_caption,
            tokenizer,
            text_encoder,
            text_projection,
            image_embeddings,
            img_ids,
            device,
            top_k=k
        )
        correct_image_id = ground_truth.get(query_caption)
        if correct_image_id in retrieved_img_ids:
            correct_retrievals += 1

    recall_at_k = correct_retrievals / len(queries) if queries else 0
    return recall_at_k


# ------------------------
# Main Function
# ------------------------

def main():
    try:
        # Initialize models
        image_encoder, text_encoder, image_projection, text_projection, cross_attention = initialize_models(DEVICE)

        # Load checkpoint
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint file not found at '{CHECKPOINT_PATH}'.")
        print("Loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        image_projection.load_state_dict(checkpoint['model_states']['image_projection'])
        text_projection.load_state_dict(checkpoint['model_states']['text_projection'])
        text_encoder.load_state_dict(checkpoint['model_states']['text_encoder'])
        cross_attention.load_state_dict(checkpoint['model_states']['cross_attention'])
        image_projection.eval()
        text_projection.eval()
        text_encoder.eval()
        cross_attention.eval()
        print("Checkpoint loaded successfully.")

        # Load data
        print(f"Loading dataset from: {DATA_PATH}")
        data = load_from_disk(DATA_PATH)
        test_data = data['test']
        print(f"Dataset loaded with {len(test_data)} samples.")

        # Load image embeddings
        if not os.path.exists(EMBEDDINGS_PATH):
            raise FileNotFoundError(f"Image embeddings file not found at '{EMBEDDINGS_PATH}'.")
        img_ids, image_embeddings = load_embeddings_hdf5(EMBEDDINGS_PATH)
        print(f"Loaded image embeddings for {len(img_ids)} images.")

        # Create Queries and Ground Truth
        num_samples = 100
        selected_data = test_data.select(range(num_samples))
        queries = []
        ground_truth = {}
        for item in selected_data:
            img_id = item['img_id']
            captions = item['caption']
            for caption in captions:
                queries.append(caption)
                ground_truth[caption] = img_id

        print(f"Created {len(queries)} query-caption pairs for evaluation.")

        # Calculate Recall@k
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        recall_k = calculate_recall_at_k(
            queries,
            ground_truth,
            tokenizer,
            text_encoder,
            text_projection,
            cross_attention,
            image_embeddings,
            img_ids,
            DEVICE,
            k=5
        )
        print(f"Recall@5: {recall_k:.4f}")

        # User Input and Retrieval
        query_caption = input("Enter a caption to search for similar images: ")
        retrieved_img_ids = retrieve_images(
            query_caption,
            tokenizer,
            text_encoder,
            text_projection,
            image_embeddings,
            img_ids,
            DEVICE,
            top_k=5
        )
        print(f"Retrieved Image IDs for query '{query_caption}': {retrieved_img_ids}")
        fetch_and_plot_images(retrieved_img_ids, test_data)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


# ------------------------
# Entry Point
# ------------------------

if __name__ == "__main__":
    main()
