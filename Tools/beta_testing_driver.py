import papermill as pm
from pathlib import Path
from datetime import datetime

# --------------------
# CONFIGURATION
# --------------------
DISEASE = "BIPOLAR"  # You can change this to test other diseases
beta_list = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.8, 0.9]

input_nb = Path("Method/beta_sensitivity_test.ipynb")
log_dir = Path(f"logs/beta_testing/{DISEASE}")
log_dir.mkdir(parents=True, exist_ok=True)

# Create a subfolder for the "junk" notebooks Papermill generates
temp_nb_dir = log_dir / "temp_notebooks"
temp_nb_dir.mkdir(exist_ok=True)

for beta in beta_list:
    log_file_path = log_dir / f"beta_{beta}.log"
    # This is the notebook Papermill must create; we put it in the temp folder
    temp_output_nb = temp_nb_dir / f"temp_{beta}.ipynb"
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running Beta {beta}... (Log: {log_file_path.name})")

    # We open the log file and tell Python to send all output there
    with open(log_file_path, "w", encoding="utf-8") as f:
        try:
            pm.execute_notebook(
                input_path=str(input_nb),
                output_path=str(temp_output_nb),
                parameters={
                    "DISEASE": DISEASE,
                    "beta": beta 
                },
                # log_output=True sends cell outputs to the console (stdout)
                log_output=True, 
                stdout_file=f,  # Redirects notebook stdout to our log file
                stderr_file=f,  # Redirects notebook errors to our log file
                cwd=str(input_nb.parent)
            )
        except Exception as e:
            f.write(f"\nFATAL ERROR during execution: {str(e)}")
            print(f"  ! Error in run {beta}. Check the log.")

print("\nAll runs complete. Logs are available in:", log_dir)