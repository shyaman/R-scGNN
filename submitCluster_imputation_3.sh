mkdir npyImputeG2E_LK_3
mkdir npyImputeR2E_LK_3
mkdir npyImputeG2E_LB_3
mkdir npyImputeR2E_LB_3

for i in {9..13}
do
sbatch run_experimentImpute_2_g_e_LK_$i.sh
sleep 1
sbatch run_experimentImpute_2_r_e_LK_$i.sh
sleep 1
sbatch run_experimentImpute_2_g_e_LB_$i.sh
sleep 1
sbatch run_experimentImpute_2_r_e_LB_$i.sh
sleep 1
done
