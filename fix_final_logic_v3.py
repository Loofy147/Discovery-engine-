import re

with open('discovery_engine_v5.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "low = raw.lower().strip()" in line:
        new_lines.append(line)
        new_lines.append("\n    # AIMO specific patterns (Olympiad level)\n")
        new_lines.append("    aimo_kws = ('triangle', 'perimeter', 'remainder', 'function f', 'tournament', 'runners', 'blackboard', 'n-tastic', 'sweets', 'norwegian', 'rectangles', 'shifty')\n")
        new_lines.append("    if any(kw in low for kw in aimo_kws) or (('integer' in low or 'integer' in raw) and not any(skip in low for skip in ('sum', 'series', 'matrix', 'graph'))):\n")
        new_lines.append("        return Problem(raw=raw, ptype=PT.AIMO)\n")
    else:
        new_lines.append(line)

with open('discovery_engine_v5.py', 'w') as f:
    f.writelines(new_lines)
