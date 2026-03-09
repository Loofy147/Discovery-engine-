import re

with open('discovery_engine_v5.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "res = next((v_ for k_, v_ in prob._cache.items() if 'summation' in k_ and v_ is not None), None)" in line:
        new_lines.append("        res = next((v_ for k_, v_ in prob._cache.items() if 'summation' in str(k_) and v_ is not None), None)\n")
    else:
        new_lines.append(line)

with open('discovery_engine_v5.py', 'w') as f:
    f.writelines(new_lines)
