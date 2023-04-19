sinteractive -A kazemian-h --gpus-per-node=1 -t 12:00:00

module load anaconda/2020.11-py38
conda activate DPLClass

# python -W ignore scGNN-rgae.py --datasetName GSE138852 --datasetDir ./  --outputDir outputdir/ --EM-iteration 5 --Regu-epochs 5 --EM-epochs 5 --quickmode --nonsparseMode --useGAEembedding --GAEepochs 30 --debugMode loadPrune
# python -W ignore scGNN-rgae.py --datasetName GSE138852 --datasetDir ./  --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 5 --EM-epochs 5 --quickmode --nonsparseMode --useGAEembedding --GAEepochs 10

# python -W ignore scGNN.py --datasetName GSE138852 --datasetDir ./  --outputDir outputdir/ --EM-iteration 5 --Regu-epochs 5 --EM-epochs 5 --quickmode --nonsparseMode --useGAEembedding --GAEepochs 30 --debugMode loadPrune
# python -W ignore scGNN.py --datasetName GSE138852 --datasetDir ./  --outputDir outputdir/ --EM-iteration 2 --Regu-epochs 5 --EM-epochs 5 --quickmode --nonsparseMode --useGAEembedding --GAEepochs 10

# (cd rgae/rgmmvgae/ && python main_cora.py)



# benchmark data process
python Preprocessing_benchmark.py --inputfile Data/benchmarkData/Chung/T2000_expression.txt --outputfile Data/benchmarkData/Chung/Chung.csv --split space --cellheadflag False --cellcount 317

python Preprocessing_benchmark.py --inputfile Data/benchmarkData/Kolodziejczyk/T2000_expression.txt --outputfile Data/benchmarkData/Kolodziejczyk/Kolodziejczyk.csv --split space --cellheadflag False --cellcount 704

python Preprocessing_benchmark.py --inputfile Data/benchmarkData/Klein/T2000_expression.txt --outputfile Data/benchmarkData/Klein/Klein.csv --split space --cellheadflag False --cellcount 2717

python Preprocessing_benchmark.py --inputfile Data/benchmarkData/Zeisel/T2000_expression.txt --outputfile Data/benchmarkData/Zeisel/Zeisel.csv --split space --cellheadflag False --cellcount 3005

python Preprocessing_main.py --expression-name Chung --featureDir Data/benchmarkData/Chung/
python Preprocessing_main.py --expression-name Kolodziejczyk --featureDir Data/benchmarkData/Kolodziejczyk/
python Preprocessing_main.py --expression-name Klein --featureDir Data/benchmarkData/Klein/
python Preprocessing_main.py --expression-name Zeisel --featureDir Data/benchmarkData/Zeisel/

# benchmark run

python3 -W ignore main_benchmark.py --datasetName Chung --benchmark Data/benchmarkData/Chung/Chung_cell_label.csv --LTMGDir Data/benchmarkData/ --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir outputDir_gpu/ --debuginfo  

python3 -W ignore main_benchmark.py --datasetName Kolodziejczyk --benchmark Data/benchmarkData/Kolodziejczyk/Kolodziejczyk_cell_label.csv --LTMGDir Data/benchmarkData/ --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir outputDir_gpu/ --debuginfo

python3 -W ignore main_benchmark.py --datasetName Klein --benchmark Data/benchmarkData/Klein/Klein_cell_label.csv --LTMGDir Data/benchmarkData/ --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir outputDir_gpu/ --debuginfo

python3 -W ignore main_benchmark.py --datasetName Zeisel --benchmark Data/benchmarkData/Zeisel/Zeisel_cell_label.csv --LTMGDir Data/benchmarkData/ --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir outputDir_gpu/ --debuginfo


python3 -W ignore main_benchmark_rgvae.py --datasetName Chung --benchmark Data/benchmarkData/Chung/Chung_cell_label.csv --LTMGDir Data/benchmarkData/ --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir outputDir_gpu_rgvae/ --debuginfo  

python3 -W ignore main_benchmark_rgvae.py --datasetName Kolodziejczyk --benchmark Data/benchmarkData/Kolodziejczyk/Kolodziejczyk_cell_label.csv --LTMGDir Data/benchmarkData/ --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir outputDir_gpu_rgvae/ --debuginfo

python3 -W ignore main_benchmark_rgvae.py --datasetName Klein --benchmark Data/benchmarkData/Klein/Klein_cell_label.csv --LTMGDir Data/benchmarkData/ --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir outputDir_gpu_rgvae/ --debuginfo

python3 -W ignore main_benchmark_rgvae.py --datasetName Zeisel --benchmark Data/benchmarkData/Zeisel/Zeisel_cell_label.csv --LTMGDir Data/benchmarkData/ --regulized-type LTMG --EMtype celltypeEM --clustering-method LouvainK --useGAEembedding --npyDir outputDir_gpu_rgvae/ --debuginfo


sbatch benchmark-Zeisel.sub
sbatch benchmark-Kolodziejczyk.sub 
sbatch benchmark-Klein.sub
sbatch benchmark-Chung.sub