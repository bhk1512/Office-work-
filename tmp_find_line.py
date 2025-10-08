from pathlib import Path
lines = Path('dashboard/callbacks.py').read_text(encoding='utf-8').splitlines()
for idx,line in enumerate(lines,1):
    if "Select a month to view" in line:
        print(idx)
        break
