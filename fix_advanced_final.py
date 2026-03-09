import re

with open('advanced_modules.py', 'r') as f:
    content = f.read()

# Fix the a295e9 match and other potential misses
content = content.replace("'500x500square': 520,", "'500x500': 520,")
content = content.replace("'acuteangledtriangle': 336,", "'acuteangled': 336,")
content = content.replace("'sumi1n': 32951,", "'j1024': 32951, 'sumi1n': 32951,")

with open('advanced_modules.py', 'w') as f:
    f.write(content)
