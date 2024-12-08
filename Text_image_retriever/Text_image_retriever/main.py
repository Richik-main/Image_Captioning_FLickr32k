import torch
from data.dataset_loader import load_data
from data.transforms import get_image_transform
from models.projection_head import ProjectionHead
from models.Crossattention import CrossAttentionModule
from transformers import ViTModel, AutoTokenizer, AutoModel
from training.trainer import train
from datasets import load_from_disk
import os
from utils.utils import compute_and_save_embeddings, save_checkpoint, load_checkpoint

# ------------------------
# Configuration
# ------------------------

EMBEDDING_DIM = 768
PROJECTION_DIM = 512
NUM_EPOCHS = 2
NUM_HEADS = 4
CHECKPOINT_PATH = 'best_model_checkpoint_modular.pth'
EMBEDDINGS_PATH = 'best_embeddings.pt'  # Path to save image embeddings
#TEXT_EMBEDDINGS_PATH = 'best_text_embeddings.pt'    # Path to save text embeddings
DATA_PATH = "/home/ubuntu/Training_img_cap/Caption_module/flickr30k_dataset_backup"

# Detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ------------------------
# Model Initialization
# ------------------------

def initialize_models(device):
    print("Loading models...")
    # Load pre-trained models
    image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
    text_encoder = AutoModel.from_pretrained('bert-base-uncased').to(device)
    image_projection = ProjectionHead(EMBEDDING_DIM, PROJECTION_DIM).to(device)
    text_projection = ProjectionHead(EMBEDDING_DIM, PROJECTION_DIM).to(device)
    #cross_attention_module = CrossAttentionModule(embed_dim=PROJECTION_DIM, num_heads=NUM_HEADS).to(device)

    # Freeze all layers in image_encoder and text_encoder
    for param in image_encoder.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    # Unfreeze the last two layers of image_encoder
    for param in image_encoder.encoder.layer[-2:].parameters():
        param.requires_grad = True
    # Unfreeze the last two layers of text_encoder
    for param in text_encoder.encoder.layer[-2:].parameters():
        param.requires_grad = True

    print("Models loaded and layers frozen/unfrozen successfully.")
    return image_encoder, text_encoder, image_projection, text_projection



# ------------------------
# Optimizer Initialization
# ------------------------

def initialize_optimizer(models, learning_rate=1e-4):
    print("Initializing optimizer...")
    parameters = []
    for model in models:
        parameters += [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    print("Optimizer initialized with parameters that require gradients.")
    return optimizer



# ------------------------
# Data Preparation
# ------------------------

def load_dataset(data_path):
    print(f"Loading dataset from: {data_path}")
    data = load_from_disk(data_path)
    print("Dataset loaded successfully.")
    return data


# ------------------------
# Main Function
# ------------------------

def main():
    try:
        # Load models
        image_encoder, text_encoder, image_projection, text_projection = initialize_models(DEVICE)

        # Initialize optimizer
        optimizer = initialize_optimizer([image_encoder, text_encoder, image_projection,
                                          text_projection])

        # Load data
        data = load_dataset(DATA_PATH)
        test_data = data['test']  # Using the test split
        print(f"Test dataset size: {len(test_data)}")

        # Tokenizer and transforms
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        transform = get_image_transform()

        # Create DataLoader
        data_loader = load_data(test_data, tokenizer, transform)
        print(f"DataLoader created with {len(data_loader)} batches.")

        # Prepare model dictionary
        model_dict = {
            'image_encoder': image_encoder,
            'text_encoder': text_encoder,
            'image_projection': image_projection,
            'text_projection': text_projection,
            #'cross_attention': cross_attention
        }

        # Check if checkpoint exists
        if os.path.exists(CHECKPOINT_PATH):
            print(f"Checkpoint found at {CHECKPOINT_PATH}. Loading checkpoint...")
            checkpoint = load_checkpoint(CHECKPOINT_PATH, model_dict, optimizer, device=DEVICE)
            start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
            best_loss = checkpoint['best_loss']
            best_model_states = checkpoint['model_states']
            print(f"Resuming training from epoch {start_epoch} with best loss {best_loss}")
        else:
            print("No checkpoint found. Starting training from scratch.")
            start_epoch = 0
            best_loss = float('inf')
            best_model_states = None

        # Train
        print("Starting training...")
        best_model_states = train(
            model_dict,
            data_loader,
            optimizer,
            DEVICE,
            NUM_EPOCHS,
            CHECKPOINT_PATH,
            start_epoch=start_epoch,
            best_loss=best_loss,
            best_model_states=best_model_states
        )
        print("Training completed successfully.")

        # Load the best model states into the models
        for k in model_dict.keys():
            model_dict[k].load_state_dict(best_model_states[k])

        print("Computing and saving embeddings...")
        compute_and_save_embeddings(
            model_dict,
            data_loader,
            DEVICE,
            EMBEDDINGS_PATH
            #TEXT_EMBEDDINGS_PATH,
            #save_every=100  # Save every 100 batches to avoid memory overflow
        )
        print("Best embeddings saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

# ------------------------
# Script Entry Point
# ------------------------

if __name__ == "__main__":
    main()