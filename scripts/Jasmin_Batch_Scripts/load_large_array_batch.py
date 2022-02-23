import subprocess

max_time = "00:10:00"
expected_time = "00:05:00"

job_name = f"load_array"
python_script_path = f"/home/users/carter10/Antarctica_Climate_Variability/scripts/large_numpy_array_load.py"

subprocess.call(["sbatch",
                        "-p", "test",
                        f"--time={max_time}",
                        f"--time-min={expected_time}",
                        "--mem=100000",
                        f"--job-name={job_name}",
                        f"--output=/home/users/carter10/job_outputs/{job_name}.out",
                        f"--error=/home/users/carter10/job_outputs/{job_name}.err",
                        python_script_path,
                        ])