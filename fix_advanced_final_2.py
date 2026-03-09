with open('advanced_modules.py', 'r') as f:
    content = f.read()
content = content.replace("'500x500': 520,", "'500times500': 520,")
with open('advanced_modules.py', 'w') as f:
    f.write(content)
