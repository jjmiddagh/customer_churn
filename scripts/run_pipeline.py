import subprocess
import config


# Execute each script
for script in config.SCRIPTS_TO_RUN:
    print(f"\nRunning: {script}")
    result = subprocess.run(['python', script], capture_output=True, text=True)

    # Print the output of each script
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error in {script}:")
        print(result.stderr)
        break
