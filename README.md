# Human-Protein-Interactions-Analysis
# Human PPI Network Analysis with Graph Theory & Machine Learning

> Structural and functional analysis of the human protein–protein interaction (PPI) network from STRING v12.0, combining classical graph theory metrics, community detection (Louvain), node embeddings (Node2Vec) and supervised link prediction (Logistic Regression / Random Forest).

Academic project for the courses **Grafos e Redes Complexas (GRC)** and **Introdução às Redes Neuronais (IRN)** 

---

## Table of Contents

- [Motivation](#motivation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [References](#references)
- [Author](#author)

---

## Motivation

Protein–protein interactions are the backbone of cellular function: signalling cascades, metabolic pathways, gene regulation. Representing the proteome as a graph (proteins = nodes, interactions = edges) lets us use the toolkit of network science to identify central proteins, detect functional modules and, ultimately, predict interactions that have not yet been observed experimentally.

This project answers the following research question:

> *"How do the structural properties of the human protein–protein interaction network allow us to identify central proteins and relevant functional modules, and to what extent does that same structure support the prediction of novel interactions not yet documented experimentally?"*

---

## Dataset

| File | Description |
| --- | --- |
| `9606.protein.links.detailed.v12.0.txt.gz` | Pairwise interactions with seven sub-scores and a combined confidence score in `[0, 1000]`. |
| `9606.protein.info.v12.0.txt.gz`           | Per-protein metadata: STRING ID, preferred gene symbol, length, functional annotation. |

Source: [STRING v12.0](https://string-db.org/cgi/download), *Homo sapiens* (taxon 9606).

After preprocessing (de-duplication, self-loop removal, confidence filter), the working graph contains **17 995 nodes** and **358 696 edges** — the *Largest Connected Component* (LCC) holds ≈ 99.6 % of all nodes.

---

## Methodology

The pipeline is built in Python with `networkx`, `node2vec`, `python-louvain` and `scikit-learn`:

1. **Data ingestion & cleaning** — load STRING dumps, filter, deduplicate, map IDs to gene symbols.
2. **Graph construction** — undirected weighted graph `G = (V, E, w)` with `w = combined_score`.
3. **Macro analysis** — density, connected components, degree distribution, distance metrics, small-world coefficient, assortativity, power-law fit.
4. **Meso analysis** — global & local clustering, triangle distribution, ER null-model comparison, community detection (Louvain), embedding-based clustering (Node2Vec + K-Means).
5. **Micro analysis** — Degree, Eigenvector, Betweenness, PageRank centralities; top-protein ranking.
6. **Link prediction** — Node2Vec embeddings + Hadamard features fed to Logistic Regression and Random Forest classifiers.
7. **Robustness simulation** — random vs. targeted attack curves on the LCC.
8. **GNN discussion** — comparative analysis of GCN / GraphSAGE / GAT / VGAE / R-GCN for extending the pipeline, with a PyTorch Geometric skeleton.

---

## Key Results

| Layer | Metric | Value |
| --- | --- | --- |
| Macro | Nodes / Edges                          | 17 995 / 358 696 |
| Macro | Density                                | ≈ 0.0022 |
| Macro | LCC coverage                           | ≈ 99.6 % |
| Macro | Average path length (sampled)          | ≈ 3.5 |
| Macro | Power-law exponent γ (MLE)             | ≈ 2.1 |
| Macro | Small-world coefficient σ              | ≈ 100 |
| Meso  | Global clustering C                    | 0.287 (≈ 130× ER baseline) |
| Meso  | Mean local clustering ⟨Cᵢ⟩             | 0.353 |
| Meso  | Louvain modularity Q                   | ≈ 0.55–0.60 |
| Meso  | Communities detected                   | ≈ 28–30 |
| Meso  | Louvain ↔ Node2Vec+KMeans NMI          | ≈ 0.54 |
| Micro | Top hub (Degree)                       | **TP53** (k = 1086) |
| Micro | Top by Eigenvector                     | **RPS9**, **RPS18**, **RPS11** (ribosomal core) |
| Micro | Top by Betweenness                     | **TP53**, **CTNNB1**, **EGFR** |

### Link prediction performance

| Model | Acc | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression (t = 0.50) | 0.827 | 0.827 | 0.826 | 0.827 | 0.903 | 0.877 |
| Logistic Regression (t = 0.37, F1-max) | — | 0.781 | 0.900 | 0.836 | 0.903 | 0.877 |
| **Random Forest (t = 0.50)** | **0.910** | **0.917** | **0.902** | **0.909** | **0.969** | **0.968** |

The top-10 predicted interactions (score ≈ 1.0) are dominated by ZNF (zinc-finger), KRTAP (keratin-associated) and PCDHB (protocadherin) families — biologically plausible candidates for experimental validation.

### Network robustness

Random removal of 10 % of nodes barely dents the LCC (loss ≈ 11 %). Targeted removal of hubs in decreasing degree order fragments the LCC by **> 60 %** with only 5 % of nodes removed — the hallmark asymmetric resilience of scale-free networks, and the empirical basis for "drug the hub" oncology strategies.

---

## Repository Structure

```
.
├── PPI_INTERACTIONS_v2.ipynb       # full analysis notebook (English)
├── PPI_Relatorio_Final.pdf         # final report (Portuguese, 24 pp.)
├── PPI_Relatorio_Final.docx        # same report, editable
├── figures_relatorio/              # all figures generated by the notebook
│   ├── fig01_full_graph.png
│   ├── fig02_degree_distribution.png
│   ├── ...
│   └── figD_powerlaw_fit.png
└── README.md
```

> ⚠️ The raw STRING dumps (`9606.protein.links.detailed.v12.0.txt.gz`, `9606.protein.info.v12.0.txt.gz`) are not committed — download them directly from <https://string-db.org/cgi/download>.

---

## How to Run

### 1. Set up the environment

```bash
git clone https://github.com/<your-user>/ppi-network-analysis.git
cd ppi-network-analysis
python -m venv .venv && source .venv/bin/activate
pip install networkx pandas numpy scipy scikit-learn \
            python-louvain node2vec matplotlib seaborn jupyterlab
```

### 2. Download the STRING data

```bash
mkdir data && cd data
wget https://stringdb-downloads.org/download/protein.links.detailed.v12.0/9606.protein.links.detailed.v12.0.txt.gz
wget https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz
cd ..
```

### 3. Launch the notebook

```bash
jupyter lab PPI_INTERACTIONS_v2.ipynb
```

Execute the cells sequentially. The heavier steps (Node2Vec training, Betweenness centrality) take several minutes on a single CPU.

### Optional: run the GNN skeleton

The final section contains a PyTorch Geometric skeleton (GAE/GCN) for link prediction. To execute it:

```bash
pip install torch torch_geometric
```

A GPU is recommended for the full training loop.

---

## Future Work

- **Real GNN implementation** — start with VGAE for self-supervised link prediction, then move to GraphSAGE with ESM-2 protein-sequence embeddings as node features.
- **Heterogeneous graphs** — protein–drug–disease tripartite networks with R-GCN, with applications to drug repurposing.
- **Temporal analysis** — successive STRING releases to study network growth and empirical preferential attachment.
- **Biological validation** — enrichment of detected communities against GO / KEGG / Reactome to confirm module-function correspondence.

---

## References

- Barabási, A.-L., & Oltvai, Z. N. (2004). *Network biology: understanding the cell's functional organization*. Nature Reviews Genetics, 5(2), 101–113.
- Albert, R., Jeong, H., & Barabási, A.-L. (2000). *Error and attack tolerance of complex networks*. Nature, 406(6794), 378–382.
- Watts, D. J., & Strogatz, S. H. (1998). *Collective dynamics of small-world networks*. Nature, 393(6684), 440–442.
- Blondel, V. D., et al. (2008). *Fast unfolding of communities in large networks*. J. Stat. Mech., 2008(10), P10008.
- Grover, A., & Leskovec, J. (2016). *node2vec: Scalable feature learning for networks*. KDD 2016.
- Kipf, T. N., & Welling, M. (2017). *Semi-supervised classification with graph convolutional networks*. ICLR 2017.
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive representation learning on large graphs*. NeurIPS 30.
- Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). *Power-law distributions in empirical data*. SIAM Review, 51(4), 661–703.
- Szklarczyk, D., et al. (2021). *The STRING database in 2021*. Nucleic Acids Research, 49(D1), D605–D612.
- Zitnik, M., Agrawal, M., & Leskovec, J. (2018). *Modeling polypharmacy side effects with graph convolutional networks*. Bioinformatics, 34(13), i457–i466.

---

## Author
| Rodrigo Sousa  
---
