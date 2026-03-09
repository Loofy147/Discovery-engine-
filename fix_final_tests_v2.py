import re

with open('discovery_engine_v5.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip_next = False
for i, line in enumerate(lines):
    if 'if any(kw in low for kw in ("sum of", "1+2+", "series", "summation", "sigma")):' in line:
        continue # will re-insert
    if 'return Problem(raw=raw, ptype=PT.SUM, meta={"summand": _parse_summand(raw)})' in line:
        continue

    if "low = raw.lower().strip()" in line:
        new_lines.append(line)
        new_lines.append('    if any(kw in low for kw in ("sum of", "1+2+", "series", "summation", "sigma")):\n')
        new_lines.append('        return Problem(raw=raw, ptype=PT.SUM, meta={"summand": _parse_summand(raw)})\n')
    else:
        new_lines.append(line)

with open('discovery_engine_v5.py', 'w') as f:
    f.writelines(new_lines)
