import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('ff640d4b'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    lines = src.split('\n')

    # Find start: line with `def _f(ar, _Tmin=`
    start_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith('def _f(ar, _Tmin=')),
        None
    )
    # Find end: the continuation line after `print(f'    -> ar_opt=`
    print_idx = next(
        (i for i, ln in enumerate(lines) if "print(f'    -> ar_opt=" in ln),
        None
    )
    if start_idx is None or print_idx is None:
        print(f'WARNING: markers not found (start={start_idx}, print={print_idx})')
        break

    # The f-string continuation sits on the very next line
    end_idx = print_idx + 1 if (
        print_idx + 1 < len(lines) and "peak_Q=" in lines[print_idx + 1]
    ) else print_idx

    # Re-indent the block by 4 spaces and wrap in try/except
    block = lines[start_idx : end_idx + 1]
    indented = ['    ' + ln if ln else ln for ln in block]

    new_lines = (
        lines[:start_idx]
        + ['            try:']
        + indented
        + ["            except Exception as e:",
           "                print(f'    ERROR ({type(e).__name__}): {e} — skipping')"]
        + lines[end_idx + 1:]
    )

    cell['source'] = '\n'.join(new_lines)
    cell['outputs'] = []
    cell['execution_count'] = None
    print(f'Wrapped lines {start_idx}–{end_idx} in try/except')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
