module purge
module load anaconda3/2023.3
conda create -n "build" python=3.10.10
conda activate build

pip install -e .

python build_datasets/build_composite_job_scripts.py
python build_datasets/build_square2d_job_scripts.py
python build_datasets/build_disk_job_scripts.py

chmod u=rwx build_datasets/run_composite_jobs.sh
chmod u=rwx build_datasets/run_square2d_jobs.sh
chmod u=rwx build_datasets/run_disk_jobs.sh

./build_datasets/run_composite_jobs.sh
./build_datasets/run_square2d_jobs.sh
./build_datasets/run_disk_jobs.sh
