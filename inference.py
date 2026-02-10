
import argparse
import torch
import pandas as pd
import re
from transformers import AutoTokenizer
from model import SOTAModel  

def heavy_clean(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = re.sub(r'[^\w\s\?\!]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_inference(model_path, input_csv, output_csv, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(input_csv)
    df['cleaned'] = df['Original_Message'].apply(heavy_clean)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SOTAModel(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for text in df['cleaned']:
            enc = tokenizer(
                text,
                max_length=192,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            logits = model(
                enc['input_ids'].to(device),
                enc['attention_mask'].to(device)
            )
            preds.append(torch.argmax(logits, dim=1).item())

    label_map = {0: 'NON_EXTREMIST', 1: 'EXTREMIST'}
    df['Prediction'] = [label_map[p] for p in preds]
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_name", default="microsoft/deberta-v3-large")
    args = parser.parse_args()

    run_inference(
        args.model_path,
        args.input_csv,
        args.output_csv,
        args.model_name
    )
