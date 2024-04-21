import os
import numpy as np
import json
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import requests
from llm_api import get_llm_response
from plot_kg import get_plot

st.set_page_config(layout='wide')

embeds_path = '../Data/embeddings'


def load_embeds(path):
    embeds = np.load(path)
    return embeds


def load_maps(path):
    with open(path, 'r') as f:
        maps = json.load(f)
    return maps


gene_embeddings = load_embeds(os.path.join(embeds_path, 'gene_embeddings.npy'))
drug_embeddings = load_embeds(os.path.join(embeds_path, 'drug_embeddings.npy'))
disease_embeddings = load_embeds(os.path.join(embeds_path, 'disease_embeddings.npy'))

custom_gene_embeddings = load_embeds(os.path.join(embeds_path, 'custom_gene_embeddings_2.npy'))
custom_drug_embeddings = load_embeds(os.path.join(embeds_path, 'custom_drug_embeddings_2.npy'))
custom_disease_embeddings = load_embeds(os.path.join(embeds_path, 'custom_disease_embeddings_2.npy'))

gene_idx_map = load_maps(os.path.join(embeds_path, 'gene_idx_map.json'))
drug_idx_map = load_maps(os.path.join(embeds_path, 'drug_idx_map.json'))
disease_idx_map = load_maps(os.path.join(embeds_path, 'disease_idx_map.json'))

gene_reverse_map = {v: k for k, v in gene_idx_map.items()}
drug_reverse_map = {v: k for k, v in drug_idx_map.items()}
disease_reverse_map = {v: k for k, v in disease_idx_map.items()}

sample_data = pickle.load(open('../sample_data.pkl', 'rb'))


def fetch_embeddings(entity, entity_type, model='base'):
    if model == 'custom':
        if entity_type == 'unknown':
            if entity in gene_idx_map:
                entity_type = 'gene'
            elif entity in drug_idx_map:
                entity_type = 'drug'
            elif entity in disease_idx_map:
                entity_type = 'disease'
        if entity_type == 'gene':
            return custom_gene_embeddings[gene_idx_map[entity]]
        elif entity_type == 'drug':
            return custom_drug_embeddings[drug_idx_map[entity]]
        elif entity_type == 'disease':
            return custom_disease_embeddings[disease_idx_map[entity]]
    else:
        if entity_type == 'gene':
            return gene_embeddings[gene_idx_map[entity]]
        elif entity_type == 'drug':
            return drug_embeddings[drug_idx_map[entity]]
        elif entity_type == 'disease':
            return disease_embeddings[disease_idx_map[entity]]


def get_triplet_embeddings(triplets):
    embeddings = []
    for triplet in triplets:
        emb1 = fetch_embeddings(triplet[0], 'unknown', model='custom')
        emb2 = fetch_embeddings(triplet[1], 'unknown', model='custom')
        triple_emb = np.mean([emb1, emb2], axis=0)
        embeddings.append(triple_emb)
    embeddings = np.array(embeddings)
    return embeddings


def get_similar_entities(entity_embedding, model='base', k=10):
    similar_genes, similar_drugs, similar_diseases = {}, {}, {}
    if model == 'custom':
        gene_similarity = cosine_similarity(entity_embedding.reshape(1, -1), custom_gene_embeddings)
        drug_similarity = cosine_similarity(entity_embedding.reshape(1, -1), custom_drug_embeddings)
        disease_similarity = cosine_similarity(entity_embedding.reshape(1, -1), custom_disease_embeddings)
    else:
        gene_similarity = cosine_similarity(entity_embedding.reshape(1, -1), gene_embeddings)
        drug_similarity = cosine_similarity(entity_embedding.reshape(1, -1), drug_embeddings)
        disease_similarity = cosine_similarity(entity_embedding.reshape(1, -1), disease_embeddings)

    for i in np.argsort(gene_similarity)[0][-k:]:
        similar_genes[gene_reverse_map[i]] = gene_similarity[0][i]
    for i in np.argsort(drug_similarity)[0][-k:]:
        similar_drugs[drug_reverse_map[i]] = drug_similarity[0][i]
    for i in np.argsort(disease_similarity)[0][-k:]:
        similar_diseases[disease_reverse_map[i]] = disease_similarity[0][i]

    return similar_genes, similar_drugs, similar_diseases


def plot_embeddings(selected_entity, entities, types, model):
    embeddings = [fetch_embeddings(entity, entity_type, model) for entity, entity_type in zip(entities, types)]
    pca = PCA(n_components=3)
    X = pca.fit_transform(embeddings)
    # tsne = TSNE(n_components=3)
    # X = tsne.fit_transform(embeddings)

    df = pd.DataFrame({'entity': entities, 'type': types, 'x': X[:, 0], 'y': X[:, 1], 'z': X[:, 2]})
    df.loc[df['entity'] == selected_entity, 'type'] = 'selected'
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='type', hover_data=['entity'])
    # legend
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def get_entity_type(entity):
    if entity in gene_idx_map:
        return 'gene'
    elif entity in drug_idx_map:
        return 'drug'
    elif entity in disease_idx_map:
        return 'disease'


def create_prompt(entities, query):
    print(entities)
    embeddings = requests.post('http://127.0.0.1:8004/embed', json={'entities': json.dumps(entities)}).json()
    embeddings = np.array(embeddings['embeddings'])
    similar_entities = {}
    all_relevant_triplets = []
    for i, entity in enumerate(entities):
        similar_genes, similar_drugs, similar_diseases = get_similar_entities(embeddings[i], k=5,
                                                                              model='custom')

        entities = list(similar_genes.keys()) + list(similar_drugs.keys()) + list(similar_diseases.keys())
        types = ['gene'] * len(similar_genes) + ['drug'] * len(similar_drugs) + ['disease'] * len(similar_diseases)

        ents = dict(zip(entities, types))

        triplets = requests.post('http://127.0.0.1:8003/get_triplets', json={'entities': json.dumps(ents)}).json()[
            'triplets']
        triplet_embeddings = get_triplet_embeddings(triplets)
        similarity = cosine_similarity(embeddings[i].reshape(1, -1), triplet_embeddings)
        top_k_indexes = np.argsort(similarity)[0][-200:]
        top_k_triplets = [triplets[i] for i in top_k_indexes]
        top_k_triplets = [triplet[0] + f'(entity_type: {get_entity_type(triplet[0])})' + ' | interacts with | ' +
                    triplet[1] + f'(entity_type: {get_entity_type(triplet[1])})' for triplet in top_k_triplets]
        triplet_embeddings = requests.post('http://127.0.0.1:8004/embed', json={'entities': json.dumps(top_k_triplets)}).json()
        triplet_embeddings = np.array(triplet_embeddings['embeddings'])
        query_embedding = requests.post('http://127.0.0.1:8004/embed', json={'entities': json.dumps([query])}).json()
        query_embedding = np.array(query_embedding['embeddings'])
        similarity = cosine_similarity(query_embedding.reshape(1, -1), triplet_embeddings)
        top_k_indexes = np.argsort(similarity)[0][-50:]
        top_k_triplets = [top_k_triplets[i] for i in top_k_indexes]
        all_relevant_triplets.extend(top_k_triplets)
    return all_relevant_triplets


if __name__ == '__main__':

    tabs = st.tabs(["Visualization", "Masked Token Prediction", "QA"])

    with tabs[0]:
        st.title('Entity Visualizer')
        entity_type = st.sidebar.selectbox('Entity Type', ['gene', 'drug', 'disease'])

        if entity_type == 'gene':
            entity = st.sidebar.selectbox('Gene', list(gene_idx_map.keys()))
        elif entity_type == 'drug':
            entity = st.sidebar.selectbox('Drug', list(drug_idx_map.keys()))
        elif entity_type == 'disease':
            entity = st.sidebar.selectbox('Disease', list(disease_idx_map.keys()))

        if st.sidebar.button("Search"):
            entity_embedding = fetch_embeddings(entity, entity_type, model='custom')
            similar_genes, similar_drugs, similar_diseases = get_similar_entities(entity_embedding, k=50,
                                                                                  model='custom')

            sorted_custom = sorted(list(similar_genes.items()), key=lambda x: x[1], reverse=True)
            sorted_custom.extend(sorted(list(similar_drugs.items()), key=lambda x: x[1], reverse=True))
            sorted_custom.extend(sorted(list(similar_diseases.items()), key=lambda x: x[1], reverse=True))

            sorted_custom = sorted(sorted_custom, key=lambda x: x[1], reverse=True)[:5]

            entities = list(similar_genes.keys()) + list(similar_drugs.keys()) + list(similar_diseases.keys())
            types = ['gene'] * len(similar_genes) + ['drug'] * len(similar_drugs) + ['disease'] * len(similar_diseases)


            custom_fig = plot_embeddings(entity, entities, types, model='custom')
            base_fig = plot_embeddings(entity, entities, types, model='base')

            cols = st.columns(2, gap="small")
            cols[0].subheader('Custom Model')
            cols[0].plotly_chart(custom_fig)
            cols[1].subheader('Base Model')
            cols[1].plotly_chart(base_fig)

            entity_embedding = fetch_embeddings(entity, entity_type, model='base')
            similar_genes, similar_drugs, similar_diseases = get_similar_entities(entity_embedding, k=50,
                                                                                  model='base')
            sorted_base = sorted(list(similar_genes.items()), key=lambda x: x[1], reverse=True)
            sorted_base.extend(sorted(list(similar_drugs.items()), key=lambda x: x[1], reverse=True))
            sorted_base.extend(sorted(list(similar_diseases.items()), key=lambda x: x[1], reverse=True))

            sorted_base = sorted(sorted_base, key=lambda x: x[1], reverse=True)[:5]
            cols = st.columns(2, gap="small")

            with cols[0]:
                with st.expander('Custom Model'):
                    for ent, score in sorted_custom:
                        st.markdown(f"{ent} (score: {score})")
            with cols[1]:
                with st.expander('Base Model'):
                    for ent, score in sorted_base:
                        st.markdown(f"{ent} (score: {score})")


    with tabs[1]:
        st.title('Masked Token Prediction')
        select_chain = st.selectbox('Chain', list(sample_data))

        ents = [ent for ent in select_chain if ent != 'interacts_with']
        masked_ent = ents[np.random.randint(0, len(ents))]
        if st.button("Predict"):

            response_custom = requests.post(
                'http://127.0.0.1:8000/predict',
                json={'text': json.dumps(select_chain), 'model': 'custom', 'masked_ent': masked_ent}
            ).json()

            response_base = requests.post(
                'http://127.0.0.1:8000/predict',
                json={'text': json.dumps(select_chain), 'model': 'base', 'masked_ent': masked_ent}
            ).json()

            for response, model_name in [(response_custom, 'Custom Model'), (response_base, 'Base Model')]:
                with st.expander(f"{model_name}"):
                    completed_text = response['text']
                    masked_ent = response['masked_ent']
                    predicted = response['best_token']

                    st.markdown("**Original Text**")
                    st.markdown(' <sep> '.join(list(select_chain)).replace(masked_ent, f':red[**{masked_ent}**]'))
                    st.write("**Completed Text(best prediction)**")
                    st.write(
                        ' <sep> '.join(tuple(completed_text.split('[SEP]'))).replace(predicted,
                                                                                     f':red[**{predicted}**]'))

    with tabs[2]:
        st.title('Question Answering')
        input = st.text_input('Question')
        if st.button("Answer"):
            entities = requests.post('http://127.0.0.1:8005/get_ner', json={'text': input}).json()['entities']
            entities = json.loads(entities)
            entities = list(entities.keys())
            st.markdown(f"Identified entities: **{entities}**")
            txt_box = st.empty()
            start_text = ''
            txt_box.markdown(start_text)
            all_relevant_triplets = create_prompt(entities, input)
            triplet_context = '\n'.join(all_relevant_triplets)
            input_prompt = "Look at the context below and answer the question asked in the best way possible based on the relevant triples retrieved from the knowledge graph:\n" + \
                           "Also provide a step by step explanation of the answer, make sure the answer is only based on the context provided, if there is no relevant information in the context, just answer with \"I don't know\"\n\n" + \
                           "Question: " + input + '\n\n' + \
                           "Given Entity: " + input + '\n\n' + \
                           "Relevant Triples:\n" + \
                           triplet_context + '\n\n' + 'Answer: '
            client = get_llm_response(input_prompt)
            for event in client.events():
                if event.data == "[DONE]":
                    break

                partial_result = json.loads(event.data)
                token = partial_result["choices"][0]["text"]
                start_text += token + ''
                txt_box.markdown(start_text)
            kg_fig = get_plot(all_relevant_triplets)

            st.plotly_chart(kg_fig, )
