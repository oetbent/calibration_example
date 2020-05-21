import subprocess


# specify paths
binary_path = "/model/IDM/EMOD/build/x64/Release/Eradication/Eradication"


# commission job
subprocess.call( [binary_path, "-C", "config.json", "-I", "/Demographics/"] )