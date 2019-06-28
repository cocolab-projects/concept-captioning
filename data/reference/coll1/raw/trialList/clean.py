import json
from pathlib import Path

""" Quick script to remove Worker IDs from Trial List.
"""

pathlist = Path('./').glob('**/*.json')
for path in pathlist:
     path = str(path)
     with open(path, 'r') as f:
         x = json.load(f)
     for round in x:
         round.pop('roles', None)
     with open(path, 'w') as f:
         json.dump(x, f)
