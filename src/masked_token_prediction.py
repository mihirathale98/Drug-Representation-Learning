from transformers import BertTokenizer, BertForMaskedLM
import torch
from fastapi import FastAPI
import numpy as np
import json
app = FastAPI()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

custom_model = BertForMaskedLM.from_pretrained('../checkpoint-44000')
model.eval()


@app.post("/predict")
def predict_mask(request: dict):
    text = json.loads(request['text'])
    model_name = request['model']
    ents = [ent for ent in text if ent != 'interacts_with']
    text = tokenizer.sep_token.join(text)
    masked_ent = ents[np.random.randint(0, len(ents))]
    text = text.replace(masked_ent, '[MASK]')

    tokenized_text = tokenizer.encode_plus(text, return_tensors="pt")

    masked_index = tokenized_text.input_ids[0].tolist().index(tokenizer.mask_token_id)

    with torch.no_grad():
        if model_name == 'custom':
            outputs = custom_model(**tokenized_text)
        else:
            outputs = model(**tokenized_text)



    masked_token_logits = outputs.logits[0, masked_index]

    top_k = 5
    probs, predicted_indices = torch.topk(torch.softmax(masked_token_logits, dim=0), top_k)
    probs = probs.tolist()
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices.tolist())
    preds = {}
    for token, prob in zip(predicted_tokens, probs):
        preds[token] = prob

    preds = dict(sorted(preds.items(), key=lambda item: item[1], reverse=True))

    predicted_token = predicted_tokens[0].lstrip('##')
    completed_text = text.replace('[MASK]', predicted_token)

    response = {"text": completed_text, "preds": preds, 'masked_ent': masked_ent}
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("masked_token_prediction:app", host="0.0.0.0", port=8000, reload=True)
