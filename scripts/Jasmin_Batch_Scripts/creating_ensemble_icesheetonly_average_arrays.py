import datetime
import subprocess

print(datetime.datetime.now())

# Creating Numpy Ensemble Average Arrays from STL Decomposition Arrays

max_time = "00:05:00"
expected_time = "00:01:00"
python_script_path = f"/home/users/carter10/Antarctica_Climate_Variability/scripts/creating_ensemble_icesheetonly_average_arrays.py"

for variable in ['temperature','snowfall','melt']:
    job_name = f"ensemble_icesheet_{variable}"
    subprocess.call(["sbatch", 
                            "-p", "short-serial-4hr",
                            "--account=short4hr",
                            f"--time={max_time}",
                            f"--time-min={expected_time}",
                            "--mem=100000",
                            f"--job-name={job_name}",
                            f"--output=/home/users/carter10/job_outputs/{job_name}.out",
                            f"--error=/home/users/carter10/job_outputs/{job_name}.err",
                            python_script_path,
                            variable
                            ])