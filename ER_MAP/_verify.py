from ER_MAP.envs.randomizer import DISEASE_POOL, DIFFICULTY_TIERS, generate_ground_truth

print(f"=== {len(DISEASE_POOL)} DISEASES ===")
for d in DISEASE_POOL:
    print(f"  {d['true_disease']}")

print()
print("=== DIFFICULTY TIERS ===")
for tier in ["easy", "medium", "hard"]:
    gt = generate_ground_truth(difficulty=tier)
    p = gt["patient"]
    print(f"  {tier.upper():8s} | compliance: {p['compliance']:20s} | comm: {p['communication']:20s} | {gt['disease']['true_disease']}")

combos = 3 * 4 * 4 * 4 * 4 * 3 * 3 * 3 * 3 * 15
print(f"\nTotal unique scenario combinations: {combos:,}")
