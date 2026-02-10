import os, random, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
# 1. Configuration & Seeding
CFG = {
    'model': 'microsoft/deberta-v3-base',
    'max_len': 192,
    'batch_size': 4,
    'accum_steps': 8,
    'epochs': 5,
    'lr': 1e-5,
    'llrd_decay': 0.9,
    'awp_lr': 1e-4,
    'awp_eps': 1e-2,
    'label_smoothing': 0.05,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

# 2. Precise Data Cleaning
def heavy_clean(text):
    text = str(text).lower()
    # Remove HTML-like tags
    text = re.sub(r'<.*?>', '', text)
    # Normalize repeated characters (extremism often uses "!!!!" or "noooo")
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    # Remove excessive punctuation but keep '?' and '!' for sentiment
    text = re.sub(r'[^\w\s\?\!]', ' ', text)
    # Fix whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Dataset Class
class ExtremismDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, i):
        enc = self.tokenizer.encode_plus(
            str(self.texts[i]), add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        item = {
            'ids': enc['input_ids'].flatten(),
            'mask': enc['attention_mask'].flatten()
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

# 4. Adversarial Weight Perturbation (AWP)
class AWP:
    def __init__(self, model, optimizer, adv_lr=1e-4, adv_eps=1e-2):
        self.model, self.optimizer = model, optimizer
        self.adv_lr, self.adv_eps = adv_lr, adv_eps
        self.backup = {}
    def perturb(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and "embeddings" not in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.adv_lr * param.grad / (norm + 1e-6)
                    param.data.add_(r_at)
                    param.data = torch.min(torch.max(param.data, self.backup[name] - self.adv_eps), self.backup[name] + self.adv_eps)
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

# 5. Architecture: Weighted Layer Pooling + Multi-Sample Dropout
class SOTAModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 * i) for i in range(1, 6)])
        self.classifier = nn.Linear(1024, 2) # Large models have 1024 hidden size
        
    def forward(self, ids, mask):
        out = self.backbone(ids, attention_mask=mask)
        # Weighted Pooling of last 4 layers
        layers = out.hidden_states
        context = torch.stack([layers[-i][:, 0, :] for i in range(1, 5)]).mean(0)
        
        for i, dropout in enumerate(self.dropouts):
            if i == 0: logits = self.classifier(dropout(context))
            else: logits += self.classifier(dropout(context))
        return logits / len(self.dropouts)

# 6. Execution Pipeline
# Data Prep
train_df = pd.read_csv('/kaggle/input/digital-extremism-detection-curated-dataset/extremism_data_final.csv').dropna(subset=['Original_Message']).reset_index(drop=True)
test_df = pd.read_csv('/kaggle/input/social-media-extreme-text-dataset/test.csv').fillna('')

train_df['cleaned'] = train_df['Original_Message'].apply(heavy_clean)
test_df['cleaned'] = test_df['Original_Message'].apply(heavy_clean)
train_df['label'] = train_df['Extremism_Label'].map({'NON_EXTREMIST': 0, 'EXTREMIST': 1})

# Loaders
tokenizer = AutoTokenizer.from_pretrained(CFG['model'])
train_ds = ExtremismDataset(train_df['cleaned'].values, train_df['label'].values, tokenizer, CFG['max_len'])
train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True)

# Training Loop
model = SOTAModel(CFG['model']).to(CFG['device'])
optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=0.01)
awp = AWP(model, optimizer, adv_lr=CFG['awp_lr'], adv_eps=CFG['awp_eps'])
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * len(train_loader) * CFG['epochs']), num_training_steps=len(train_loader) * CFG['epochs'])
criterion = nn.CrossEntropyLoss(label_smoothing=CFG['label_smoothing'])

print("Starting Heavy Training...")
model.train()
for epoch in range(CFG['epochs']):
    for i, batch in enumerate(train_loader):
        ids, mask, labels = batch['ids'].to(CFG['device']), batch['mask'].to(CFG['device']), batch['labels'].to(CFG['device'])
        logits = model(ids, mask)
        loss = criterion(logits, labels) / CFG['accum_steps']
        loss.backward()
        
        if epoch >= 1: # AWP Activation
            awp.perturb()
            loss_adv = criterion(model(ids, mask), labels)
            loss_adv.backward()
            awp.restore()
            
        if (i + 1) % CFG['accum_steps'] == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    print(f"Epoch {epoch+1} Done.")

# 7. Inference
model.eval()
test_ds = ExtremismDataset(test_df['cleaned'].values, None, tokenizer, CFG['max_len'])
test_loader = DataLoader(test_ds, batch_size=CFG['batch_size'], shuffle=False)
preds = []
with torch.no_grad():
    for batch in test_loader:
        logits = model(batch['ids'].to(CFG['device']), batch['mask'].to(CFG['device']))
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

reverse_map = {0: 'NON_EXTREMIST', 1: 'EXTREMIST'}
pd.DataFrame({'ID': test_df['ID'], 'Extremism_Label': [reverse_map[p] for p in preds]}).to_csv('submission.csv', index=False)
print("Submission.csv generated successfully.")