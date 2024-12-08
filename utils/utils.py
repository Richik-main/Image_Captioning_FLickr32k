import torch
from tqdm import tqdm
import h5py

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"Checkpoint saved at {filename}")

def load_checkpoint(checkpoint_path, model_dict, optimizer, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for name, model in model_dict.items():
        model.load_state_dict(checkpoint['model_states'][name])
        model.to(device)  # Ensure the model is on the correct device

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Move optimizer state tensors to the correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    print(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint  # Return the checkpoint dictionary
# def compute_and_save_embeddings(model_dict, data_loader, device, image_embeddings_path, text_embeddings_path):
#     model_dict['image_encoder'].eval()
#     model_dict['text_encoder'].eval()
#     model_dict['image_projection'].eval()
#     model_dict['text_projection'].eval()
#
#     # Move models to the specified device
#     for key in model_dict:
#         model_dict[key] = model_dict[key].to(device)
#
#     all_image_embeddings = []
#     all_text_embeddings = []
#     all_img_ids = []
#
#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc="Computing Embeddings"):
#             img_ids, images, input_ids, attention_masks = batch
#             images = images.to(device)
#             input_ids = input_ids.to(device)
#             attention_masks = attention_masks.to(device)
#
#             # Forward pass for images
#             image_outputs = model_dict['image_encoder'](images)
#             image_embeddings = image_outputs.last_hidden_state[:, 0]  # CLS token
#             image_proj = model_dict['image_projection'](image_embeddings)
#             all_image_embeddings.append(image_proj.cpu())
#             all_img_ids.extend(img_ids)
#
#             # Forward pass for text
#             text_outputs = model_dict['text_encoder'](input_ids=input_ids, attention_mask=attention_masks)
#             text_embeddings = text_outputs.last_hidden_state[:, 0]  # CLS token
#             text_proj = model_dict['text_projection'](text_embeddings)
#             all_text_embeddings.append(text_proj.cpu())
#
#     # Concatenate all embeddings
#     all_image_embeddings = torch.cat(all_image_embeddings)
#     all_text_embeddings = torch.cat(all_text_embeddings)
#
#     # Save embeddings separately
#     torch.save({
#         'img_ids': all_img_ids,
#         'image_embeddings': all_image_embeddings,
#     }, image_embeddings_path)
#
#     torch.save({
#         'text_embeddings': all_text_embeddings,
#     }, text_embeddings_path)
#
#     print(f"Image embeddings saved to {image_embeddings_path}")
#     print(f"Text embeddings saved to {text_embeddings_path}")
#
# def compute_and_save_embeddings(model_dict, data_loader, device, image_embeddings_path, text_embeddings_path):
#     model_dict['image_encoder'].eval()
#     model_dict['text_encoder'].eval()
#     model_dict['image_projection'].eval()
#     model_dict['text_projection'].eval()
#
#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc="Computing Embeddings"):
#             img_ids, images, input_ids, attention_masks = batch
#             images = images.to(device)
#             input_ids = input_ids.to(device)
#             attention_masks = attention_masks.to(device)
#
#             # Forward pass for images
#             image_outputs = model_dict['image_encoder'](images)
#             image_embeddings = image_outputs.last_hidden_state[:, 0]  # CLS token
#             image_proj = model_dict['image_projection'](image_embeddings).cpu()
#
#             # Forward pass for text
#             text_outputs = model_dict['text_encoder'](input_ids=input_ids, attention_mask=attention_masks)
#             text_embeddings = text_outputs.last_hidden_state[:, 0]  # CLS token
#             text_proj = model_dict['text_projection'](text_embeddings).cpu()
#
#             # Save embeddings for this batch
#             batch_image_data = {
#                 'img_ids': img_ids,
#                 'image_embeddings': image_proj,
#             }
#             batch_text_data = {
#                 'text_embeddings': text_proj,
#             }
#
#             # Append to files
#             torch.save(batch_image_data, image_embeddings_path + f'_{img_ids[0]}.pt')
#             torch.save(batch_text_data, text_embeddings_path + f'_{img_ids[0]}.pt')
#
#     print(f"Image embeddings saved to {image_embeddings_path}")
#     print(f"Text embeddings saved to {text_embeddings_path}")




def compute_and_save_embeddings(model_dict, data_loader, device, embeddings_path, save_every=100):
    model_dict['image_encoder'].eval()
    model_dict['text_encoder'].eval()
    model_dict['image_projection'].eval()
    model_dict['text_projection'].eval()

    # Move models to the specified device
    for key in model_dict:
        model_dict[key] = model_dict[key].to(device)

    # Dynamically determine the embedding dimension using a dummy forward pass
    dummy_input = torch.zeros(1, model_dict['image_encoder'].config.hidden_size).to(device)
    embedding_dim = model_dict['image_projection'](dummy_input).shape[-1]

    # Open HDF5 file in appendable mode
    with h5py.File(embeddings_path, 'w') as h5_file:
        # Create datasets for image embeddings, text embeddings, and image IDs
        num_samples = len(data_loader.dataset)

        image_embeddings_ds = h5_file.create_dataset(
            'image_embeddings', shape=(num_samples, embedding_dim), dtype='float32'
        )
        text_embeddings_ds = h5_file.create_dataset(
            'text_embeddings', shape=(num_samples, embedding_dim), dtype='float32'
        )
        img_ids_ds = h5_file.create_dataset(
            'img_ids', shape=(num_samples,), dtype=h5py.string_dtype(encoding='utf-8')
        )

        start_idx = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc="Computing Embeddings")):
                img_ids, images, input_ids, attention_masks = batch
                batch_size = len(img_ids)

                images = images.to(device)
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)

                # Forward pass for images
                image_outputs = model_dict['image_encoder'](images)
                image_embeddings = image_outputs.last_hidden_state[:, 0]  # CLS token
                image_proj = model_dict['image_projection'](image_embeddings).cpu().numpy()

                # Forward pass for text
                text_outputs = model_dict['text_encoder'](input_ids=input_ids, attention_mask=attention_masks)
                text_embeddings = text_outputs.last_hidden_state[:, 0]  # CLS token
                text_proj = model_dict['text_projection'](text_embeddings).cpu().numpy()

                # Write embeddings and IDs into HDF5 file
                end_idx = start_idx + batch_size
                image_embeddings_ds[start_idx:end_idx] = image_proj
                text_embeddings_ds[start_idx:end_idx] = text_proj
                img_ids_ds[start_idx:end_idx] = [id.encode('utf8') for id in img_ids]

                start_idx = end_idx  # Update index for next batch

        print(f"Embeddings saved incrementally to {embeddings_path}")