import os
import subprocess

for model in [1, 2, 3, 4, 5]:
    subprocess.run(['python', 'get_map.py', 'for_expression/', f'{model}'])