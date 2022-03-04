import datetime
import subprocess

print(datetime.datetime.now())

# This script takes the ensemble stl decomposition array of shape [8,3,456,392,504] and samples it to a grid cell on Larsen C [235,55]

max_time = "00:15:00"
expected_time = "00:05:00"

job_name = f"larsenc_sampling"
python_script_path = f"/home/users/carter10/Antarctica_Climate_Variability/scripts/sampling_stl_decomp_larsen_c.py"

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