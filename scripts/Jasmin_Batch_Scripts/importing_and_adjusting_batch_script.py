import datetime
import subprocess

print(datetime.datetime.now())

# Importing and Adjusting Data

max_time = "04:00:00"
expected_time = "02:00:00"
python_script = f"/home/users/carter10/Antarctica_Climate_Variability/scripts/importing_and_adjusting.py"

for variable in ['temperature','snowfall','melt']:
    job_name = f"import_raw_data_and_adjust_{variable}"
    subprocess.call(["sbatch", 
                            "-p", "short-serial-4hr",
                            "--account=short4hr",
                            #"-p", "short-serial",
                            f"--time={max_time}",
                            f"--time-min={expected_time}",
                            "--mem=100000",
                            f"--job-name={job_name}",
                            f"--output=/home/users/carter10/job_outputs/{job_name}.out",
                            f"--error=/home/users/carter10/job_outputs/{job_name}.err",
                            python_script,
                            variable,
                            ])