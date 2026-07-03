content = open('run_experiments_sequential.py', encoding='utf-8').read()
old = "                cwd=REPO_ROOT,\n                capture_output=True,\n                text=True,\n                env=run_env,\n            )"
new = "                cwd=REPO_ROOT,\n                capture_output=True,\n                text=True,\n                encoding=\"utf-8\",\n                errors=\"replace\",\n                env=run_env,\n            )"
content = content.replace(old, new)
open('run_experiments_sequential.py', 'w', encoding='utf-8').write(content)
print("Fixed!")