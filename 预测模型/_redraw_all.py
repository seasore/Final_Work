# -*- coding: utf-8 -*-
import sys, os, io, warnings, json
warnings.filterwarnings('ignore')

if sys.platform == 'win32' and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from param_search import _huaTanSuoTuBiao, _huaCanShuZuiYouZongJie, PARAM_RESULT_DIR

csv_dir = PARAM_RESULT_DIR
redrawn = []
for fname in sorted(os.listdir(csv_dir)):
    if fname.startswith('param_') and fname.endswith('.csv') and 'epoch' not in fname:
        param_name = fname.replace('param_', '').replace('.csv', '')
        df = pd.read_csv(os.path.join(csv_dir, fname))
        print(f'Redrawing: {param_name}  ({len(df)} rows)')
        _huaTanSuoTuBiao(df, param_name, csv_dir)
        redrawn.append(param_name)

best_json = os.path.join(csv_dir, 'param_search_best.json')
if os.path.exists(best_json):
    with open(best_json) as f:
        best_vals = json.load(f)
    _huaCanShuZuiYouZongJie(best_vals, csv_dir)
    print('Summary table redrawn.')

print(f'Done. Redrawn: {redrawn}')
