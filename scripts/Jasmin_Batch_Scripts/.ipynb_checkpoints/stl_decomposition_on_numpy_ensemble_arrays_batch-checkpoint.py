import datetime
import subprocess

print(datetime.datetime.now())

# Creating Numpy Ensemble Arrays from Regridded Monthly Data

max_time = "02:00:00"
expected_time = "01:00:00"
python_script_path = f"/home/users/carter10/Antarctica_Climate_Variability/scripts/stl_decomposition_on_numpy_ensemble_arrays.py"
CPUs = 16

for variable in ['temperature','snowfall','melt']:
    job_name = f"stl_decomposition_{variable}"
    subprocess.call(["sbatch", 
                            f"--ntasks={CPUs}",
                            "-p", "par-single",
                            f"--time={max_time}",
                            f"--time-min={expected_time}",
                            "--mem=100000",
                            f"--job-name={job_name}",
                            f"--output=/home/users/carter10/job_outputs/{job_name}.out",
                            f"--error=/home/users/carter10/job_outputs/{job_name}.err",
                            python_script_path,
                            variable
                            ])