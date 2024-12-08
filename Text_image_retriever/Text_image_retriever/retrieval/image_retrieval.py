import torch
import re
from matplotlib import pyplot as plt
from torch.nn.functional import normalize
# ------------------------
# Preprocessing Functions
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

def retrieve_images(query_caption, tokenizer, text_encoder, text_projection, image_embeddings, img_ids, device, top_k=5):
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
    # # Get top_k results
    # top_k_indices = similarities.topk(top_k).indices
    # retrieved_img_ids = [img_ids[idx] for idx in top_k_indices]
    # return retrieved_img_ids
    # Collect unique image IDs
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