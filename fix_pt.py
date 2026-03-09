with open('discovery_engine_v5.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "PDE_RD=23" in line and "AIMO" not in line:
        new_lines.append(line.replace("PDE_RD=23", "PDE_RD=23; AIMO=24"))
    elif '99:"unknown"' in line and "24" not in line:
        new_lines.append('            24:"AIMO problem",\n' + line)
    else:
        new_lines.append(line)

with open('discovery_engine_v5.py', 'w') as f:
    f.writelines(new_lines)
