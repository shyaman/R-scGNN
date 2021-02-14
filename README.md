# scGNN   

## About:

__scGNN__ (**s**ingle **c**ell **g**raph **n**eural **n**etworks) provides a hypothesis-free deep learning framework for scRNA-Seq analyses. This framework formulates and aggregates cell-cell relationships with graph neural networks and models heterogeneous gene expression patterns using a left-truncated mixture Gaussian model. scGNN integrates three iterative multi-modal autoencoders and outperforms existing tools for gene imputation and cell clustering on four benchmark scRNA-Seq datasets.

This repository contains the source code for the paper **scGNN: a novel graph neural network framework for single-cell RNA-Seq analyses** preprint available at bioRxiv; doi: https://doi.org/10.1101/2020.08.02.233569

**BibTeX**

```latex
@article{Wang_2020_scGNN,
	author = {Juexin Wang, Anjun Ma, Yuzhou Chang, Jianting Gong, Yuexu Jiang, Hongjun Fu, Cankun Wang, Ren Qi, Qin Ma, Dong Xu},
	title = {scGNN: a novel graph neural network framework for single-cell RNA-Seq analyses},
	elocation-id = {2020.08.02.233569},
	year = {2020},
	doi = {10.1101/2020.08.02.233569},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/10.1101/2020.08.02.233569v1},
	eprint = {https://www.biorxiv.org/content/10.1101/2020.08.02.233569v1.full.pdf},
	journal = {bioRxiv}
}
```
--------------------------------------------------------------------------------

## Installation:

Installation Tested on Ubuntu 16.04 and CentOS 7 with Python 3.6.8

### From Source:

Start by grabbing this source codes:

```bash
git clone https://github.com/juexinwang/scGNN.git
cd scGNN
```

### Option 1 : (Recommended) Use python virutal environment with conda（<https://anaconda.org/>）

```shell
conda create -n scgnnEnv python=3.6.8 pip
conda activate scgnnEnv
conda install r-devtools
conda install -c r r-igraph
conda install -c cyz931123 r-scgnnltmg
pip install -r requirements.txt
```

### Option 2 : Direct install individually

Need to install R packages first, tested on R >=3.6.1:

In R command line:

```R
install.packages("devtools")
install.packages("igraph")
library(devtools)
install_github("BMEngineeR/scGNNLTMG")
```

Then install all python packages in bash.

```bash
pip install -r requirements.txt
```

### Option 3: Use Docker

Download and install [docker](https://www.docker.com/products/docker-desktop).

Pull docker image **gongjt057/scgnn** from the [dockerhub](https://hub.docker.com/). Please beware that this image is huge for it includes all the environments. To get this docker image as base image type the command as shown in the below:

```bash
docker pull gongjt057/scgnn:code
```

Type `docker images` to see the list of images you have downloaded on your machine. If **gongjt057/scgnn** in the list, download it successfully.

Run a container from the image.

```bash
docker run -it gongjt057/scgnn:code /bin/bash
cd /scGNN/scGNN
```

Then you can proceed to the next step.

## Quick Start

scGNN accepts scRNA-seq data format: CSV and 10X

### 1. Prepare datasets

#### CSV format

Take an example of Alzheimer’s disease datasets ([GSE138852](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE138852)) analyzed in the manuscript.

```shell
mkdir GSE138852
wget -P GSE138852/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE138nnn/GSE138852/suppl/GSE138852_counts.csv.gz
```

#### 10X format

Take an example of [liver cellular landscape study](https://data.humancellatlas.org/explore/projects/4d6f6c96-2a83-43d8-8fe1-0f53bffd4674) from human cell atlas(<https://data.humancellatlas.org/>)

```shell
mkdir liver
wget -P liver https://data.humancellatlas.org/project-assets/project-matrices/4d6f6c96-2a83-43d8-8fe1-0f53bffd4674.homo_sapiens.mtx.zip
cd liver
unzip 4d6f6c96-2a83-43d8-8fe1-0f53bffd4674.homo_sapiens.mtx.zip
cd ..
```

### 2. Preprocess input files

This step generates Use_expression.csv (preprocessed file) and gets discretized regulatory signals as ltmg.csv from Left-Trunctruncated-Mixed-Gaussian(LTMG) model (Optional but recommended).  

In preprocessing, parameters are used:

- **filetype** defines file type (CSV or 10X(default))  
- **geneSelectnum** selects a number of most variant genes. The default gene number is 2000 

The running time is depended on the cell number and gene numbers selected. It takes ~20 minutes (GSE138852) and ~28 minutes (liver) to generate the files needed.

#### CSV format

```shell
python -W ignore PreprocessingscGNN.py --datasetName GSE138852_counts.csv.gz --datasetDir GSE138852/ --LTMGDir GSE138852/ --filetype CSV --geneSelectnum 2000
```

#### 10X format

```shell
python -W ignore PreprocessingscGNN.py --datasetName 481193cb-c021-4e04-b477-0b7cfef4614b.mtx --datasetDir liver/ --LTMGDir liver/ --geneSelectnum 2000
```

### 3. Run scGNN

We take an example of an analysis in GSE138852. Here we use parameters to demo purposes:

- **batch-size** defines batch-size of the cells for training
- **EM-iteration** defines the number of iteration, default is 10, here we set as 2. 
- **quickmode** for bypassing cluster autoencoder.
- **Regu-epochs** defines epochs in feature autoencoder, default is 500, here we set as 50.
- **EM-epochs** defines epochs in feature autoencoder in the iteration, default is 200, here we set as 20.

If you want to reproduce results in the manuscript, do not use these parameters. 

#### CSV format

For CSV format, we need add **--nonsparseMode**

```bash
python -W ignore scGNN.py --datasetName GSE138852 --datasetDir ./ --LTMGDir ./ --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --quickmode --nonsparseMode
```

#### 10X format

```bash
python -W ignore scGNN.py --datasetName 481193cb-c021-4e04-b477-0b7cfef4614b.mtx --LTMGDir liver/ --datasetDir ./ --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 50 --EM-epochs 20 --quickmode
```

On these demo dataset using single cpu, the running time of demo codes is ~33min/26min. User should get exact same results as paper shown with full running time on single cpu for ~6 hours. If user wants to use multiple CPUs, parameter **--coresUsage** can be set as **all** or any number of cores the machine has.

### 4. Check Results

In outputdir now, we have four output files.

- ***_recon.csv**:        Imputed gene expression matrix. Row as gene, col as cell. First row as gene name, First col as the cell name. 

- ***_embedding.csv**:    Learned embedding (features) for clustering. Row as cell, col as embeddings. First row as the embedding names (no means). First col as the cell name.

- ***_graph.csv**:        Learned graph edges of the cell graph in tuples: nodeA,nodeB,weights. First row as the name.

- ***_results.txt**:      Identified cell types. First row as the name. 

For a complete list of options provided by scGNN:

```bash
python scGNN.py --help
```

More information can be checked at the [tutorial](https://github.com/juexinwang/scGNN/tree/master/tutorial).

Another repository for reviewers is located at [here](https://github.com/scgnn/scGNN/).

## Reference:

1. VAE <https://github.com/pytorch/examples/tree/master/vae>
2. GAE <https://github.com/tkipf/gae/tree/master/gae>
3. scVI-reproducibility <https://github.com/romain-lopez/scVI-reproducibility>
4. LTMG <https://academic.oup.com/nar/article/47/18/e111/5542876>

## Contact:

Juexin Wang: wangjue@missouri.edu

Anjun Ma: Anjun.Ma@osumc.edu

Qin Ma: Qin.Ma@osumc.edu

Dong Xu: xudong@missouri.edu
