# Drug Representation Learning

## Introduction
Knowledge Graphs (KGs) are invaluable tools for encapsulating large amounts of information within a manageable and structured framework. This project aims to integrate KGs into the biomedical domain, enhancing the speed and efficacy of research and experimental processes. Leveraging both a custom-built KG and advanced language models such as BERT, our system enhances biomedical question-answering capabilities and supports the discovery of novel medical treatments and therapies.

## Setup

1. Ensure Python version 3.7 or higher is installed.
2. Run `pip install -r requirements.txt` to install necessary packages.
3. Verify the correct installation of all packages.
4. Ensure availability of computational resources, preferably with GPU support, to manage deep learning models.

## Components

### Knowledge Graph Assembly
- **Dataset Integration**: Utilizes the Stanford Network Analysis Project (SNAP) dataset to form a robust KG encompassing genes, drugs, and diseases.
- **Graph Construction**: Builds a comprehensive KG from combined datasets, representing entities and their interactions through nodes and edges.

### Language Model Integration
- **BERT for Semantic Analysis**: Applies BERT to derive semantic understanding from the KG, aiding in the representation of complex biomedical entities.
- **Custom Model Training**: Focuses on adapting pre-trained models to specific needs of biomedical data processing.

### Information Retrieval System
- **Entity Recognition**: Implements Named Entity Recognition (NER) to identify biomedical entities in queries.
- **Query Processing**: Constructs queries using detected entities and KG data to fetch relevant information.
- **Similarity Search and Indexing**: Employs advanced indexing techniques to manage and retrieve KG information efficiently.

### Answer Generation
- **Language Model Application**: Utilizes a custom language model to generate accurate answers based on the processed queries.
- **Integration of NER and Language Models**: Combines NER outputs with language models to refine answer generation, ensuring relevance and precision.

## Contributors
- Mihir Athale
- Isha Singh
- Shreyas Terdalkar
- Suhaani Agarwal
