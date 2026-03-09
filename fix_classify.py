with open('discovery_engine_v5.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "low = raw.lower().strip()" in line:
        new_lines.append(line)
        new_lines.append("\n    if any(kw in low for kw in ('triangle', 'perimeter', 'remainder', 'integer', 'function f', 'tournament', 'runners')):\n")
        new_lines.append("        return Problem(raw=raw, ptype=PT.AIMO)\n")
    else:
        new_lines.append(line)

with open('discovery_engine_v5.py', 'w') as f:
    f.writelines(new_lines)
