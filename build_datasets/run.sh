module purge
module load anaconda3/2023.3
conda create -n "build" python=3.10.10
conda activate build

pip install -e .

python build_job_scripts.py
run_jobs.sh
