import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor, AutoTokenizer, AutoModel
from load_data import data
import re


# ------------------------
# Preprocessing for the captions
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


# ------------------------
# Dataset Class
# ------------------------

class PreprocessingDataset(Dataset):
    def __init__(self, dataset, transform=None, tokenizer=None, max_length=32):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform

        for sample in dataset:
            img_id = sample['img_id']
            image = sample['image']
            for caption in sample['caption']:
                self.data.append((img_id, image, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_id, image, caption = self.data[index]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Tokenize the caption
        input_id, attention_mask = preprocess_text(caption, tokenizer=self.tokenizer, max_length=self.max_length)

        return img_id, image, input_id, attention_mask


# ------------------------
# Transformations
# ------------------------

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ------------------------
# Load Tokenizer and Models
# ------------------------

# Text tokenizer and model
text_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text_encoder = AutoModel.from_pretrained('bert-base-uncased')

# Image processor and model
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False)

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_encoder.to(device)
image_encoder.to(device)
# ------------------------
# Dataset and DataLoader
# ------------------------

# Assuming 'data' is your dataset containing 'img_id', 'image', and 'caption'
dataset = PreprocessingDataset(data, transform=image_transform, tokenizer=text_tokenizer)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)