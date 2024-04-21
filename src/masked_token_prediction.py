from transformers import BertTokenizer, BertForMaskedLM
import torch
from fastapi import FastAPI
import numpy as np
import json
app = FastAPI()
from transformers import pipeline

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

custom_model = BertForMaskedLM.from_pretrained('../checkpoint-44000').eval()

@app.post("/predict")
def predict_mask(request: dict):
    text = json.loads(request['text'])
    model_name = request['model']
    masked_ent = request['masked_ent']
    text = tokenizer.sep_token.join(text)
    original_text = text
    text = text.replace(masked_ent, ''.join(['[MASK]'] * len(tokenizer.tokenize(masked_ent))))
    token_ids = tokenizer.encode(text, return_tensors='pt')

    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()

    masked_pos = [mask.item() for mask in masked_position]

    with torch.no_grad():
        if model_name == 'base':
            output = model(token_ids)[0].squeeze()
        else:
            output = custom_model(token_ids)[0].squeeze()

    masked_token_logits = output[masked_pos]

    top_1 = torch.topk(masked_token_logits, 1, dim=-1).indices.tolist()
    top_1 = [tokenizer.decode(tok, skip_special_tokens=True, clean_up_tokenization_spaces=True) for tok in top_1]

    best_token = tokenizer.convert_tokens_to_string(top_1)
    completed_text = original_text.replace(masked_ent, best_token)
    response = {"text": completed_text, 'masked_ent': masked_ent, 'best_token': best_token}

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("masked_token_prediction:app", host="0.0.0.0", port=8000, reload=True)
