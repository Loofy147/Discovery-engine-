import discovery_engine_v5 as eng
p = eng.classify('Let $ABC$ be an acute-angled triangle')
print(f"PT: {p.ptype}")
print(f"Label: {p.ptype.label()}")
