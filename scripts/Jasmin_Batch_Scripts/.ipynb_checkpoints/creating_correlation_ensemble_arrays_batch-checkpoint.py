import datetime
import subprocess

print(datetime.datetime.now())

# Creating Correlation Ensemble Arrays from STL Decomposition Monthly Data

max_time = "01:00:00"
expected_time = "00:30:00"
python_script_path = f"/home/users/carter10/Antarctica_Climate_Variability/scripts/creating_correlation_ensemble_arrays.py"

for variable in ['temperature','snowfall','melt']:
    job_name = f"correlation_{variable}"
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