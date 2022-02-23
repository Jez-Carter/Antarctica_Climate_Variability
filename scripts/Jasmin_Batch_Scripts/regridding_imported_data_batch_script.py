import datetime
import subprocess

print(datetime.datetime.now())

# Regridding Imported Data

max_time = "04:00:00"
expected_time = "02:00:00"
python_script_path = f"/home/users/carter10/Antarctica_Climate_Variability/scripts/regridding_imported_data.py"

for variable in ['temperature','snowfall','melt']:
    job_name = f"regrid_imported_data{variable}"
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
                            variable,
                            ])

