import os
import sys
import runpy

# Determine base paths
current_dir = os.path.dirname(os.path.abspath(__file__))
inner_dir = os.path.join(current_dir, "cost-sensitive-aps-classifier")
target_script = os.path.join(inner_dir, "app.py")

# Ensure inner directory is in sys.path
if inner_dir not in sys.path:
    sys.path.insert(0, inner_dir)

if os.path.exists(target_script):
    runpy.run_path(target_script, run_name="__main__")
else:
    raise FileNotFoundError(f"Could not locate inner app script at: {target_script}")
