"""Quick script to regenerate reward curve from saved eval_results.json"""
import json
import sys
sys.path.insert(0, ".")
from ER_MAP.evaluate import plot_reward_curve

with open("d:/Meta_Finals/ER_MAP/eval_results.json") as f:
    results = json.load(f)

plot_reward_curve(results, "d:/Meta_Finals/ER_MAP/reward_curve.png")
print("Done!")
