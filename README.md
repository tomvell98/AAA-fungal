# AAA-fungal

Scripts accompanying the manuscript "Exploring Fungal Lysine Synthesis: Evolution and Diversity of the Enzymes in the Alpha-Aminoadipate Pathway Across the Fungal Kingdom"

This repository contains analysis scripts grouped in `methods/` and small configuration files in `config/`. 

## Repository layout
- `methods/` : scripts and templates for retrieval, phylogeny, conservation/structure, localization, and protein language model analyses
- `config/` : sample sets and query sequences used in the study

## Environment (conda)
Create the environment:

```bash
conda env create -f environment.yml
conda activate aaa-fungal

## External programs 

Required:
- BUSCO v5.8.2
- InterPro release 102.0  
- MMseqs2 version 18.8cc5c
- MAFFT version 7.526
- trimAl version v1.5.rev0
- IQ-TREE version 3.0.1
- Docker / container (for AlphaFold3)
