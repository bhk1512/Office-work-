from pathlib import Path
p=Path('dashboard/callbacks.py')
s=p.read_text(encoding='utf-8')
s=s.replace('(mode_value or toggle_value or "erection")','(toggle_value or mode_value or "erection")')
p.write_text(s, encoding='utf-8')
print('replaced')