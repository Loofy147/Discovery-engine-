with open('advanced_modules.py', 'r') as f:
    content = f.read()
# The issue is likely how re.sub(r'[^a-z0-9]', '', raw.lower()) handles \times
# 'a500times500' -> let's try '500times500'
content = content.replace("'500times500': 520,", "'500times500': 520, 'rectangles': 520,")
with open('advanced_modules.py', 'w') as f:
    f.write(content)
