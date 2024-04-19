import json

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
from fastapi import FastAPI
from functools import lru_cache

app = FastAPI()

@lru_cache(maxsize=None)
def init_model(model_name, checkpoint=None):
    if checkpoint is None:
        checkpoint = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(checkpoint).to(device)
    return tokenizer, model

tokenizer, model = init_model('bert-base-uncased','../checkpoint-44000')

batch_size=32

@app.post("/embed")
def get_embeddings(request: dict):
    text_list = json.loads(request['entities'])
    embeddings = np.array([])
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch = text_list[i:i+batch_size]
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        attn_mask = encoded_input['attention_mask'].cpu()
        with torch.no_grad():
            model_output = model(**encoded_input)[0].squeeze(0)
        embeds = model_output.cpu()
        # take mean of embeddings of all tokens in each sentence in the batch use attn_mask to ignore padding
        embeds = (embeds * attn_mask.unsqueeze(2)).sum(1) / attn_mask.sum(1).unsqueeze(1)
        if embeddings.size == 0:
            embeddings = embeds
        else:
            embeddings = np.concatenate((embeddings, embeds), axis=0)
    return {'embeddings': embeddings.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("entity_embedder:app", host="0.0.0.0", port=8004, reload=True)
