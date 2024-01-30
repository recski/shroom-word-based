import json
import sys

data = json.load(sys.stdin)

sorted_data = sorted(data, key=lambda e: e["id"])

json.dump(sorted_data, sys.stdout)
