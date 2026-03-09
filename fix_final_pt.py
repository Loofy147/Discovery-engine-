import re

with open('discovery_engine_v5.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "PDE_RD=23" in line and "AIMO" not in line:
        new_lines.append(line.replace("PDE_RD=23", "PDE_RD=23; AIMO=24"))
    elif '99:"unknown"' in line and "24" not in line:
        new_lines.append('            24:"AIMO problem",\n' + line)
    elif "low = raw.lower().strip()" in line:
        new_lines.append(line)
        new_lines.append("\n    if any(kw in low for kw in ('triangle', 'perimeter', 'remainder', 'integer', 'function f', 'tournament', 'runners', 'what is', 'solve', 'calculate')):\n")
        new_lines.append("        return Problem(raw=raw, ptype=PT.AIMO)\n")
    else:
        new_lines.append(line)

with open('discovery_engine_v5.py', 'w') as f:
    f.writelines(new_lines)
