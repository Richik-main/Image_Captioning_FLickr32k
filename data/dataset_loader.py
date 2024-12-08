import re
from torch.utils.data import Dataset, DataLoader

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


class PreprocessingDataset(Dataset):
    def __init__(self, dataset, transform=None, tokenizer=None, max_length=32):
        self.data = []
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        for sample in dataset:
            img_id = sample['img_id']
            image = sample['image']
            for caption in sample['caption']:
                self.data.append((img_id, image, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_id, image, caption = self.data[index]
        if self.transform:
            image = self.transform(image)
        input_id, attention_mask = preprocess_text(caption, self.tokenizer, self.max_length)
        return img_id, image, input_id, attention_mask


def load_data(data, tokenizer, transform, batch_size=32):
    dataset = PreprocessingDataset(data, transform=transform, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

