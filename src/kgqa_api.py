import pandas as pd
from fastapi import FastAPI
import json
from functools import lru_cache
from tqdm import tqdm

app = FastAPI()

# Load DataFrames once and cache the results
@lru_cache(maxsize=None)
def load_dataframes():
    return {
        'drug_gene_df': pd.read_csv('../Data/preprocessed_data/drug_gene.csv'),
        'disease_gene_df': pd.read_csv('../Data/preprocessed_data/disease_gene.csv'),
        'disease_drug_df': pd.read_csv('../Data/preprocessed_data/disease_drug.csv'),
        'drug_drug_df': pd.read_csv('../Data/preprocessed_data/drug_drug.csv')
    }

def get_gene_triplets(gene, drug_gene_df, disease_gene_df):
    triplets = []
    triplets += drug_gene_df[drug_gene_df['gene_name'] == gene].values.tolist()
    triplets += disease_gene_df[disease_gene_df['gene_name'] == gene].values.tolist()
    return triplets

def get_drug_triplets(drug, disease_drug_df, drug_gene_df, drug_drug_df):
    triplets = []
    triplets += disease_drug_df[disease_drug_df['drug_name'] == drug].values.tolist()
    triplets += drug_gene_df[drug_gene_df['drug_name'] == drug].values.tolist()
    triplets += drug_drug_df[(drug_drug_df['drug_1_name'] == drug) | (drug_drug_df['drug_2_name'] == drug)].values.tolist()
    return triplets

def get_disease_triplets(disease, disease_drug_df, disease_gene_df):
    triplets = []
    triplets += disease_drug_df[disease_drug_df['disease_name'] == disease].values.tolist()
    triplets += disease_gene_df[disease_gene_df['disease_name'] == disease].values.tolist()
    return triplets

@app.post("/get_triplets")
def get_triplets(request: dict):
    relevant_entities = json.loads(request['entities'])
    dfs = load_dataframes()  # Load or retrieve DataFrames from cache

    relevant_triplets = []

    for entity, type in relevant_entities.items():
        if type == 'gene':
            relevant_triplets.extend(get_gene_triplets(entity, dfs['drug_gene_df'], dfs['disease_gene_df']))
        elif type == 'drug':
            relevant_triplets.extend(get_drug_triplets(entity, dfs['disease_drug_df'], dfs['drug_gene_df'], dfs['drug_drug_df']))
        elif type == 'disease':
            relevant_triplets.extend(get_disease_triplets(entity, dfs['disease_drug_df'], dfs['disease_gene_df']))

    relevant_triplets = [list(x) for x in set(tuple(x) for x in relevant_triplets)]
    return {'triplets': relevant_triplets}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("kgqa_api:app", host="0.0.0.0", port=8003, reload=True)
