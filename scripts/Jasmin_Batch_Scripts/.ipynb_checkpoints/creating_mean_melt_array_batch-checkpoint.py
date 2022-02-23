import datetime
import subprocess

print(datetime.datetime.now())

# Creating Mean Melt Array from STL Decomposition Monthly Data

max_time = "00:15:00"
expected_time = "00:05:00"

job_name = f"mean_melt"
python_script_path = f"/home/users/carter10/Antarctica_Climate_Variability/scripts/creating_mean_melt_array.py"

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
                        ])