import networkx as nx
import pandas as pd
import plotly.graph_objects as go

def get_plot(triplets):
    triplets = [triplet.split('interacts with') for triplet in triplets]

    head = [triplet[0].strip() for triplet in triplets]
    relation = ['interacts with' for triplet in triplets]
    tail = [triplet[1].strip() for triplet in triplets]

    df = pd.DataFrame({'head': head, 'relation': relation, 'tail': tail})

    # Create a graph
    G = nx.Graph()
    for _, row in df.iterrows():
      G.add_edge(row['head'], row['tail'], label=row['relation'])

    # pos = nx.spring_layout(G)
    pos = nx.kamada_kawai_layout(G)

    # pos = nx.fruchterman_reingold_layout(G, k=0.5)
    # Create edge trace
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=0.5, color='gray'),
            hoverinfo='none'
         )
        edge_traces.append(edge_trace)

    # Create node trace
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        marker=dict(size=10, color='lightblue'),
        text=[node.split('(')[0] for node in G.nodes()],
        textposition='top center',
        hoverinfo='text',
        textfont=dict(size=12)
    )

    # node_labels_traces = go.Scatter(
    #     x=[pos[node][0] for node in G.nodes()],
    #     y=[pos[node][1] for node in G.nodes()],
    #     mode='text',
    #     text=[node.split('(')[0] for node in G.nodes()],
    #     textposition='middle center',
    #     hoverinfo='none',
    #     textfont=dict(size=7)
    # )

    # Create edge label trace
    # edge_label_trace = go.Scatter(
    #     x=[(pos[edge[0]][0] + pos[edge[1]][0]) / 2 for edge in G.edges()],
    #     y=[(pos[edge[0]][1] + pos[edge[1]][1]) / 2 for edge in G.edges()],
    #     mode='text',
    #     text=[G[edge[0]][edge[1]]['label'] for edge in G.edges()],
    #     textposition='middle center',
    #     hoverinfo='none',
    #     textfont=dict(size=7)
    # )

    # Create layout
    layout = go.Layout(
        title='Knowledge Graph',
        titlefont_size=16,
        title_x=0.5,
        showlegend=False,
        hovermode='closest',
        xaxis_visible=False,
        yaxis_visible=False,
        width=1000,
    )

    # make traces off by default
    fig = go.Figure(data=edge_traces + [node_trace], layout=layout)
    fig.update_traces(visible='legendonly')


    return fig

