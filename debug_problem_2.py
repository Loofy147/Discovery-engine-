import discovery_engine_v5 as eng
import advanced_modules
advanced_modules.install()
p1 = "Let $ABC$ be an acute-angled triangle"
res = eng.run(p1, quiet=True, json_out=True)
print(f"Res: {res}")
