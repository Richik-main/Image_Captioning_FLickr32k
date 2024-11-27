import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from natsort import natsorted
from text_embeddings import tokenizer, model  # Assuming you have a tokenizer and model defined
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # Import tqdm for progress bars

# ----------------------
# Data loader
# ----------------------

class ImageCaptionDataset(Dataset):
    def __init__(self, file_paths, batch_size_per_file):
        """
        Initialize the dataset with file paths and batch size per file.
        """
        self.file_paths = file_paths
        self.batch_size_per_file = batch_size_per_file
        self.index_map = []  # Global index mapping to (file_index, sample_slice)

        # Build an index mapping (global index -> (file_idx, sample_slice))
        for file_idx, file_path in enumerate(file_paths):
            # Load the file once to get the number of samples
            file_data = torch.load(file_path)
            num_samples = len(file_data)
            # Create slices for batching
            for start_idx in range(0, num_samples, batch_size_per_file):
                end_idx = min(start_idx + batch_size_per_file, num_samples)
                sample_slice = slice(start_idx, end_idx)
                self.index_map.append((file_idx, sample_slice))

    def __len__(self):
        """Return the total number of batches across all files."""
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_slice = self.index_map[idx]
        file_path = self.file_paths[file_idx]
        file_data = torch.load(file_path)

        batch_data = file_data[sample_slice]

        # Prepare the batch
        img_ids = [sample["img_id"] for sample in batch_data]
        image_embeddings = torch.stack([sample["image_embedding"] for sample in batch_data])
        text_embeddings = torch.stack([sample["text_embedding"] for sample in batch_data])
        token_ids = torch.stack([sample["token_id"] for sample in batch_data])
        attention_masks = torch.stack([sample["attention_mask"] for sample in batch_data])

        # Ensure that image_embeddings have shape [batch_size, seq_len_img, embed_dim]
        if len(image_embeddings.shape) == 2:
            # Add a sequence length dimension if missing
            image_embeddings = image_embeddings.unsqueeze(1)  # [batch_size, 1, embed_dim]

        return (
            img_ids,            # List of image IDs
            image_embeddings,   # Tensor [batch_size_per_file, seq_len_img, embed_dim]
            text_embeddings,    # Tensor [batch_size_per_file, seq_len_text, embed_dim]
            token_ids,          # Tensor [batch_size_per_file, seq_len_text]
            attention_masks,    # Tensor [batch_size_per_file, seq_len_text]
        )

# Directory for saved embeddings
save_dir = "embeddings_final/train"

# Get a sorted list of all batch files
batch_files = natsorted(glob.glob(f"{save_dir}/train_batch_*.pt"))
print(f"Found batch files: {batch_files}")

# Split files into training and validation sets
train_files = batch_files[:10]  # First 10 files for training
val_files = batch_files[10:12]  # Next 2 files for validation

# Define batch size per file
batch_size_per_file = 10  # Adjust based on your GPU's memory capacity

# Create dataset instances
train_dataset = ImageCaptionDataset(train_files, batch_size_per_file)
val_dataset = ImageCaptionDataset(val_files, batch_size_per_file)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

# -----------------------
# TransformerEncoderLayer
# -----------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        # Self-Attention block
        x_norm = self.layer_norm_1(x)
        attn_output, _ = self.self_attention(x_norm, x_norm, x_norm)
        x = x + attn_output

        # Feed-Forward Network
        x_norm = self.layer_norm_2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        return x

# -----------------------
# TransformerDecoderLayer
# -----------------------

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, units, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.layernorm_3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, units),
            nn.ReLU(),
            nn.Linear(units, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, encoder_output, self_attention_mask=None, attn_mask=None):
        # Padding mask for self-attention
        key_padding_mask = None
        if self_attention_mask is not None:
            key_padding_mask = self_attention_mask == 0  # True where padding tokens exist

        # Causal mask for autoregressive decoding
        if attn_mask is None:
            seq_len = inputs.size(1)  # Length of the input sequence
            attn_mask = torch.triu(
                torch.ones((seq_len, seq_len), device=inputs.device), diagonal=1
            ).bool()  # Upper triangular mask

        # Self-Attention
        x_norm = self.layernorm_1(inputs)
        attn_output, _ = self.self_attention(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,  # Causal mask
            key_padding_mask=key_padding_mask  # Padding mask
        )
        x = inputs + attn_output

        # Cross-Attention
        x_norm = self.layernorm_2(x)
        attn_output, _ = self.cross_attention(
            x_norm, encoder_output, encoder_output,
            key_padding_mask=None  # Adjust if encoder outputs have padding
        )
        x = x + attn_output

        # Feed-Forward Network
        x_norm = self.layernorm_3(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)
        return x

# -----------------------
# TransformerEncoder
# -----------------------

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers=6, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim, num_heads) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x_out = self.layer_norm(x)
        return x_out

# -----------------------
# TransformerDecoder
# -----------------------

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, units, num_heads, num_layers=6, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, units, dropout) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)  # Optional: Final layer norm
        self.vocab_projection = nn.Linear(embed_dim, vocab_size)

    def forward(self, inputs, encoder_output, self_attention_mask=None, attn_mask=None, use_embeddings=True):
        if use_embeddings:
            x = inputs  # Inputs are embeddings
        else:
            x = self.embedding(inputs)  # Convert token IDs to embeddings

        for layer in self.layers:
            x = layer(x, encoder_output, self_attention_mask, attn_mask)

        x = self.layer_norm(x)  # Optional: Apply final layer norm
        logits = self.vocab_projection(x)  # [batch_size, seq_len_text, vocab_size]
        return logits

# -----------------------
# ImageCaptioningModel
# -----------------------

class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image_embeddings, inputs, attention_mask=None, use_embeddings=True):
        """
        Forward pass of the image captioning model.

        :param image_embeddings: Precomputed image embeddings [batch_size, seq_len_img, embed_dim].
        :param inputs: Precomputed text embeddings or token IDs [batch_size, seq_len_text].
        :param attention_mask: Self-attention mask for the decoder [batch_size, seq_len_text].
        :param use_embeddings: If True, inputs are embeddings; else, inputs are token IDs.
        :return: Decoder output [batch_size, seq_len_text, vocab_size].
        """
        # Encode image embeddings
        encoder_output = self.encoder(image_embeddings)

        # Decode with cross-attention
        decoder_output = self.decoder(
            inputs,
            encoder_output,
            self_attention_mask=attention_mask,  # Self-attention mask for text
            attn_mask=None,
            use_embeddings=use_embeddings
        )
        return decoder_output

# ----------------------
# Training & Validation
# ----------------------

def train_one_epoch_with_dataloader(model, optimizer, train_loader, device, loss_fn, tokenizer, accumulation_steps=8):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler()

    # Add tqdm progress bar
    train_loader_tqdm = tqdm(train_loader, desc="Training", unit="batch")

    for step, (img_ids, image_embeddings, text_embeddings, token_ids, attention_mask) in enumerate(train_loader_tqdm):
        # Since batch_size=1 in DataLoader and Dataset returns batches, remove the extra dimension
        img_ids = img_ids[0]
        image_embeddings = image_embeddings[0]
        input_embeddings = text_embeddings[0]
        token_ids = token_ids[0]
        attention_mask = attention_mask[0]

        # Move data to device
        image_embeddings = image_embeddings.to(device)  # Shape: [batch_size, seq_len_img, embed_dim]
        input_ids = token_ids.to(device).long()         # Shape: [batch_size, seq_len_text]
        attention_mask = attention_mask.to(device).float()  # Shape: [batch_size, seq_len_text]

        # Shift input_ids for input and target
        input_ids_in = input_ids[:, :-1]  # Remove the last token for inputs
        target_tokens = input_ids[:, 1:]  # Remove the first token for targets
        attention_mask_in = attention_mask[:, :-1]  # Adjust attention mask for input length

        # Debug: Print shapes on the first iteration
        if step == 0:
            print(f"image_embeddings shape: {image_embeddings.shape}")
            print(f"input_ids_in shape: {input_ids_in.shape}")
            print(f"target_tokens shape: {target_tokens.shape}")
            print(f"attention_mask_in shape: {attention_mask_in.shape}")

        # Forward pass
        with torch.cuda.amp.autocast():
            decoder_output = model(
                image_embeddings,
                inputs=input_ids_in,
                attention_mask=attention_mask_in,  # This is the padding mask
                use_embeddings=False  # Use token IDs
            )
            loss = loss_fn(decoder_output.transpose(1, 2), target_tokens) / accumulation_steps

        # Backpropagation
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

        # Update tqdm description
        train_loader_tqdm.set_postfix(loss=total_loss / (step + 1))

    return total_loss / len(train_loader)


def validate_one_epoch_with_dataloader_autoregressive(
    model, val_loader, device, tokenizer, loss_fn, vocab_size, max_caption_length=50
):
    """
    Validation loop with autoregressive inference for caption generation.

    :param model: The trained image captioning model.
    :param val_loader: DataLoader for validation data.
    :param device: The device (CPU/GPU) to run the model on.
    :param tokenizer: Tokenizer to decode token IDs into text.
    :param loss_fn: Loss function for evaluation.
    :param vocab_size: Size of the vocabulary.
    :param max_caption_length: Maximum length of generated captions.
    :return: Average validation loss and sample captions for debugging.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    generated_samples = []

    # Add tqdm progress bar
    val_loader_tqdm = tqdm(val_loader, desc="Validation", unit="batch")
    min_caption_length = 5

    with torch.no_grad():
        for step, (img_ids, image_embeddings, text_embeddings, token_ids, attention_mask) in enumerate(val_loader_tqdm):
            # Since batch_size=1 in DataLoader and Dataset returns batches of size=10, remove the extra dimension
            img_id = img_ids[0]
            image_embedding = image_embeddings[0].to(device)  # Shape: [batch_size, seq_len_img, embed_dim]
            text_embedding = text_embeddings[0].to(device)    # Shape: [batch_size, seq_len_text, embed_dim]
            input_ids = token_ids[0].to(device).long()        # Shape: [batch_size, seq_len_text]
            attention_mask = attention_mask[0].to(device).float()  # Shape: [batch_size, seq_len_text]

            # Determine if the sample has ground truth captions
            is_train = attention_mask.sum().item() > 0

            if is_train:
                # Shift input_ids for loss computation
                input_ids_in = input_ids[:, :-1]
                target_tokens = input_ids[:, 1:]
                attention_mask_loss = attention_mask[:, :-1]

                # Initialize autoregressive decoding with <sos> token
                batch_size = image_embedding.size(0)
                generated_tokens = torch.full(
                    (batch_size, 1), tokenizer.cls_token_id, device=device, dtype=torch.long
                )

                # Encode image embeddings once
                encoder_output = model.encoder(image_embedding)

                # Autoregressive decoding
                for t in range(max_caption_length):
                    seq_len = generated_tokens.size(1)
                    causal_mask = torch.triu(
                        torch.ones((seq_len, seq_len), device=device), diagonal=1
                    ).bool()

                    # Decoder forward pass using token IDs
                    decoder_output = model.decoder(
                        generated_tokens,
                        encoder_output,
                        attn_mask=causal_mask,
                        use_embeddings=False
                    )

                    # Predict the next token using top-k sampling
                    next_token_logits = decoder_output[:, -1, :]
                    temperature = 1.0
                    next_token_logits = next_token_logits / temperature
                    top_k = 5
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k, dim=-1)
                    next_token_id = torch.multinomial(top_k_probs, 1)
                    next_token_id = top_k_indices.gather(-1, next_token_id)

                    # Append the predicted token
                    generated_tokens = torch.cat([generated_tokens, next_token_id], dim=1)

                    # Stop if all sequences generate <eos> and minimum length is reached
                    if t >= min_caption_length:
                        if (next_token_id == tokenizer.sep_token_id).all():
                            break

                # Compute loss with teacher forcing
                decoder_output = model(
                    image_embedding,
                    inputs=input_ids_in,
                    attention_mask=attention_mask_loss,
                    use_embeddings=False
                )

                # Compute loss
                loss = loss_fn(decoder_output.transpose(1, 2), target_tokens)
                total_loss += loss.item()
                total_samples += 1

                # Update tqdm description
                val_loader_tqdm.set_postfix(loss=total_loss / total_samples)

                # Decode target and generated captions
                if len(generated_samples) < 5:
                    target_caption = tokenizer.decode(input_ids[0, 1:].tolist(), skip_special_tokens=True)
                    generated_caption = tokenizer.decode(generated_tokens[0, 1:].tolist(), skip_special_tokens=True)
                    generated_samples.append((img_id, target_caption, generated_caption))

                    # Print debug information for the first few samples
                    if step < 1:
                        print(f"\nImage ID: {img_id}")
                        print(f"Target Caption (Ground Truth): {target_caption}")
                        print(f"Generated Caption (Model Output): {generated_caption}")

            else:
                # Test data without ground truth captions
                # Initialize autoregressive decoding with <sos> token
                batch_size = image_embedding.size(0)
                generated_tokens = torch.full(
                    (batch_size, 1), tokenizer.cls_token_id, device=device, dtype=torch.long
                )

                # Encode image embeddings once
                encoder_output = model.encoder(image_embedding)

                # Autoregressive decoding
                for t in range(max_caption_length):
                    seq_len = generated_tokens.size(1)
                    causal_mask = torch.triu(
                        torch.ones((seq_len, seq_len), device=device), diagonal=1
                    ).bool()

                    # Decoder forward pass using token IDs
                    decoder_output = model.decoder(
                        generated_tokens,
                        encoder_output,
                        attn_mask=causal_mask,
                        use_embeddings=False
                    )

                    # Predict the next token using top-k sampling
                    next_token_logits = decoder_output[:, -1, :]
                    temperature = 1.0
                    next_token_logits = next_token_logits / temperature
                    top_k = 5
                    next_token_probs = F.softmax(next_token_logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(next_token_probs, top_k, dim=-1)
                    next_token_id = torch.multinomial(top_k_probs, 1)
                    next_token_id = top_k_indices.gather(-1, next_token_id)

                    # Append the predicted token
                    generated_tokens = torch.cat([generated_tokens, next_token_id], dim=1)

                    # Stop if all sequences generate <eos> and minimum length is reached
                    if t >= min_caption_length:
                        if (next_token_id == tokenizer.sep_token_id).all():
                            break

                # Decode and store the generated captions
                if len(generated_samples) < 5:
                    generated_caption = tokenizer.decode(generated_tokens[0, 1:].tolist(), skip_special_tokens=True)
                    generated_samples.append((img_id, None, generated_caption))

                    # Print debug information for the first few samples
                    if step < 1:
                        print(f"\nImage ID: {img_id}")
                        print(f"Generated Caption (Model Output): {generated_caption}")

        # Compute average loss only if total_samples > 0
        avg_loss = total_loss / total_samples if total_samples > 0 else None
        return avg_loss, generated_samples

# -----------------------
# Main Training and Validation Loop
# -----------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = tokenizer.vocab_size
print(f"Vocabulary Size: {vocab_size}")
embed_dim = 768
num_heads = 8
units = 2048  # Typically 4 times embed_dim in Transformers

num_encoder_layers = 6  # Increased depth
num_decoder_layers = 6  # Increased depth
dropout = 0.1

loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Initialize Multi-Layer Encoder and Decoder
encoder = TransformerEncoder(embed_dim, num_heads, num_layers=num_encoder_layers, dropout=dropout).to(device)
decoder = TransformerDecoder(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    units=units,
    num_heads=num_heads,
    num_layers=num_decoder_layers,
    dropout=dropout
).to(device)

# Load pre-trained embeddings from RoBERTa
pretrained_embeddings = model.embeddings.word_embeddings.weight.clone().detach()

assert pretrained_embeddings.size(0) == vocab_size, "Mismatch in vocabulary sizes between tokenizer and pre-trained model."

decoder.embedding.weight.data.copy_(pretrained_embeddings)

# Ensure the embedding layer is trainable
decoder.embedding.weight.requires_grad = True

# Create the final model
model_final = ImageCaptioningModel(encoder, decoder).to(device)

optimizer = torch.optim.AdamW(model_final.parameters(), lr=1e-4, weight_decay=1e-5)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training settings
best_val_loss = float("inf")
best_model_path = "best_image_captioning_model.pth"
num_epochs = 10
accumulation_steps = 16

# Main Training and Validation Loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Training phase
    train_loss = train_one_epoch_with_dataloader(
        model_final, optimizer, train_loader, device, loss_fn, tokenizer, accumulation_steps
    )
    print(f"Train Loss: {train_loss:.4f}")

    # Validation phase
    val_loss, generated_samples = validate_one_epoch_with_dataloader_autoregressive(
        model_final, val_loader, device, tokenizer, loss_fn, vocab_size
    )
    print(f"Validation Loss: {val_loss:.4f}")

    # Optionally, print a few generated samples
    for img_id, target, generated in generated_samples[:5]:
        print(f"\nImage ID: {img_id}")
        print(f"Target Caption (Ground Truth): {target}")
        print(f"Generated Caption (Model Output): {generated}")

    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model_final.state_dict(), best_model_path)
        print(f"New best model saved with validation loss: {val_loss:.4f}")
    scheduler.step()
