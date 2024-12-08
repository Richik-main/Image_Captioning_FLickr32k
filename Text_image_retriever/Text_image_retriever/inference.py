import torch
import os
from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
from models.projection_head import ProjectionHead
from retrieval.image_retrieval import retrieve_images, fetch_and_plot_images, encode_texts, preprocess_text
from torch.nn.functional import normalize
import h5py
from tqdm import tqdm

# ------------------------
# Configuration
# ------------------------

EMBEDDING_DIM = 768
PROJECTION_DIM = 512
CHECKPOINT_PATH = 'best_model_checkpoint_modular_new.pth'
DATA_PATH = "/home/ubuntu/Training_img_cap/Caption_module/flickr30k_dataset_backup"
EMBEDDINGS_PATH = "best_embeddings.pt"  # Adjusted to HDF5 file
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ------------------------
# Loading Functions
# ------------------------

def load_embeddings_hdf5(embeddings_path):
    with h5py.File(embeddings_path, 'r') as h5_file:
        img_ids = h5_file['img_ids'][:]  # Load all image IDs
        img_ids = [id.decode('utf8') for id in img_ids]  # Decode bytes to strings
        image_embeddings = torch.tensor(h5_file['image_embeddings'][:])  # Load image embeddings
    return img_ids, normalize(image_embeddings, p=2, dim=-1)

# ------------------------
# Recall@k Calculation
# ------------------------

def calculate_recall_at_k(queries, ground_truth, tokenizer, text_encoder, text_projection, image_embeddings, img_ids, device, k=5):
    correct_retrievals = 0

    for query_caption in tqdm(queries, desc="Evaluating Recall@k"):
        # Retrieve top-k unique image IDs
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

        # Check if the correct image ID is in the top-k results
        correct_image_id = ground_truth.get(query_caption)
        if correct_image_id in retrieved_img_ids:
            correct_retrievals += 1

    # Calculate Recall@k
    total_queries = len(queries)
    recall_at_k = correct_retrievals / total_queries if total_queries > 0 else 0
    return recall_at_k

# ------------------------
# Main Function
# ------------------------

def main():
    try:
        # ------------------------
        # Model Initialization
        # ------------------------
        print("Loading models...")
        text_projection = ProjectionHead(EMBEDDING_DIM, PROJECTION_DIM).to(DEVICE)
        text_encoder = AutoModel.from_pretrained('bert-base-uncased').to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Load checkpoint
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Checkpoint file not found at '{CHECKPOINT_PATH}'.")
        print("Loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        text_projection.load_state_dict(checkpoint['model_states']['text_projection'])
        text_encoder.load_state_dict(checkpoint['model_states']['text_encoder'])
        text_projection.eval()
        text_encoder.eval()
        print("Checkpoint loaded successfully.")

        # ------------------------
        # Data Loading
        # ------------------------
        print(f"Loading dataset from: {DATA_PATH}")
        data = load_from_disk(DATA_PATH)
        test_data = data['test']  # Use test data for retrieval
        print(f"Dataset loaded with {len(test_data)} samples.")

        # Load image embeddings
        if not os.path.exists(EMBEDDINGS_PATH):
            raise FileNotFoundError(f"Image embeddings file not found at '{EMBEDDINGS_PATH}'.")
        img_ids, image_embeddings = load_embeddings_hdf5(EMBEDDINGS_PATH)
        print(f"Loaded image embeddings for {len(img_ids)} images.")

        # ------------------------
        # Create Queries and Ground Truth
        # ------------------------
        # Use a subset for evaluation to save time
        num_samples = 100  # Adjust as needed
        selected_data = test_data.select(range(num_samples))

        queries = []
        ground_truth = {}
        for item in selected_data:
            img_id = item['img_id']
            captions = item['caption']  # List of captions
            for caption in captions:
                queries.append(caption)
                ground_truth[caption] = img_id

        print(f"Created {len(queries)} query-caption pairs for evaluation.")

        # ------------------------
        # Calculate Recall@k
        # ------------------------
        recall_k = calculate_recall_at_k(
            queries,
            ground_truth,
            tokenizer,
            text_encoder,
            text_projection,
            image_embeddings,
            img_ids,
            DEVICE,
            k=5
        )
        print(f"Recall@5: {recall_k:.4f}")

        # ------------------------
        # User Input and Retrieval
        # ------------------------
        query_caption = input("Enter a caption to search for similar images: ")

        # Retrieve images
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

        # Fetch and plot retrieved images
        fetch_and_plot_images(retrieved_img_ids, test_data)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

# ------------------------
# Entry Point
# ------------------------

if __name__ == "__main__":
    main()
