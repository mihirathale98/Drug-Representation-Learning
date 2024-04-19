import spacy
from fastapi import FastAPI
import json

app = FastAPI()
class NER:
    def __init__(self, model1='en_ner_bc5cdr_md', model2='en_ner_bionlp13cg_md'):
        self.model_1 = spacy.load(model1)
        self.model_2 = spacy.load(model2)

    def get_entities_model_bc5(self, document):
        doc = self.model_1(document)
        entities = {}
        for ent in doc.ents:
            entities[ent.text] = ent.label_
        return entities

    def get_entities_model_bio13(self, document):
        doc = self.model_2(document)
        entities = {}
        for ent in doc.ents:
            entities[ent.text] = ent.label_
        return entities

    def get_all_entities(self, document):
        entities_model_bc5 = self.get_entities_model_bc5(document)
        entities_model_bio13 = self.get_entities_model_bio13(document)
        all_entities_set = set(entities_model_bc5.keys()) | set(entities_model_bio13.keys())

        all_entities = {}
        for ent_, label in entities_model_bc5.items():
            all_entities[ent_] = label

        for ent_, label in entities_model_bio13.items():
            if ent_ not in all_entities:
                all_entities[ent_] = label

        return all_entities


ner = NER()


@app.post('/get_ner')
def get_entities(request: dict):
    document = request['text']
    return {'entities': json.dumps(ner.get_all_entities(document))}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("ner_api:app", host="0.0.0.0", port=8005, reload=True)