"""
Packaging and Verification Script for Version 2.3 Submission
Apex Entity Resolvers - ML Challenge 2026
"""

import os
import sys
import shutil
import zipfile
import subprocess
from collections import Counter

def main():
    base_dir = r"c:\Users\megha\Downloads\6ab10eb3b23ba_student_resource\student_resource"
    match_v23 = os.path.join(base_dir, "output", "matching_results_v23.tsv")
    cand_v23 = os.path.join(base_dir, "output", "candidate_pairs_v23.tsv")

    print("=" * 60)
    print("=== Apex Entity Resolvers: Version 2.3 Packaging ===")
    print("=" * 60)

    if not os.path.exists(match_v23):
        print(f"ERROR: {match_v23} does not exist yet!")
        sys.exit(1)

    match_sz = os.path.getsize(match_v23) / (1024 * 1024)
    cand_sz = os.path.getsize(cand_v23) / (1024 * 1024)
    print(f"Output files found:")
    print(f"  matching_results_v23.tsv: {match_sz:.2f} MB")
    print(f"  candidate_pairs_v23.tsv:  {cand_sz:.2f} MB")

    # 1. Collision and Sanity Check
    print("\n[Step 1/4] Running duplicate target collision audit on matching_results_v23.tsv...")
    total_entities = 0
    empty_singletons = 0
    total_matches = 0
    target_counts = Counter()

    with open(match_v23, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            total_entities += 1
            parts = line.strip().split("\t")
            if len(parts) < 2 or not parts[1].strip():
                empty_singletons += 1
            else:
                matches = parts[1].split(",")
                total_matches += len(matches)
                for m in matches:
                    target_counts[m] += 1

    collisions = [m for m, count in target_counts.items() if count > 1]
    singleton_pct = (empty_singletons / total_entities) * 100 if total_entities else 0.0
    avg_matches = (total_matches / total_entities) if total_entities else 0.0

    print(f"  Total Source 1 Entities: {total_entities:,}")
    print(f"  Singletons (Empty matches): {empty_singletons:,} ({singleton_pct:.2f}%)")
    print(f"  Total Matches Predicted: {total_matches:,} (Avg {avg_matches:.2f}/entity)")
    print(f"  Duplicate Target Collisions: {len(collisions)}")

    if len(collisions) > 0:
        print(f"  WARNING: Found {len(collisions)} collisions! Sample: {collisions[:5]}")
    else:
        print("  PERFECT: Exactly 0 target collisions detected! Complete mutual exclusivity verified.")

    # 2. Prepare official submission files
    print("\n[Step 2/4] Staging official matching_results.tsv and candidate_pairs.tsv...")
    dest_match = os.path.join(base_dir, "output", "matching_results.tsv")
    dest_cand = os.path.join(base_dir, "output", "candidate_pairs.tsv")
    shutil.copyfile(match_v23, dest_match)
    shutil.copyfile(cand_v23, dest_cand)
    print(f"  Copied v23 -> {dest_match}")
    print(f"  Copied v23 -> {dest_cand}")

    # 3. Run official submission validator
    print("\n[Step 3/4] Running official competition validator with --check-ids...")
    validator_path = os.path.join(base_dir, "utils", "validate_submission.py")
    test_dir = os.path.join(base_dir, "dataset", "test")
    
    cmd = [
        sys.executable,
        validator_path,
        "--matching", dest_match,
        "--candidate", dest_cand,
        "--test-dir", test_dir,
        "--check-ids"
    ]
    print(f"  Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("  Validator Output:")
    for line in res.stdout.strip().split("\n"):
        print(f"    {line}")
    if res.stderr.strip():
        print("  Validator Stderr:")
        for line in res.stderr.strip().split("\n"):
            print(f"    {line}")

    if res.returncode != 0:
        print(f"  ERROR: Validator failed with return code {res.returncode}")
        sys.exit(res.returncode)

    # 4. Packaging into submission zip
    print("\n[Step 4/4] Creating Apex_Entity_Resolvers_submission_v23.zip...")
    zip_path = os.path.join(base_dir, "Apex_Entity_Resolvers_submission_v23.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        print(f"  Adding output/matching_results.tsv...")
        zf.write(dest_match, arcname="output/matching_results.tsv")
        print(f"  Adding output/candidate_pairs.tsv...")
        zf.write(dest_cand, arcname="output/candidate_pairs.tsv")
        
        doc_path = os.path.join(base_dir, "Documentation.md")
        if os.path.exists(doc_path):
            print(f"  Adding Documentation.md...")
            zf.write(doc_path, arcname="Documentation.md")

        code_dir = os.path.join(base_dir, "code", "business_entity_resolution")
        for root, dirs, files in os.walk(code_dir):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for file in files:
                if file.endswith((".py", ".joblib", ".md", ".txt", ".json", ".yaml", ".sh")):
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, base_dir)
                    print(f"  Adding {rel_p}...")
                    zf.write(full_p, arcname=rel_p)

    final_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print("=" * 60)
    print(f"SUCCESS: Package created successfully!")
    print(f"Location: {zip_path}")
    print(f"Size: {final_size_mb:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    main()
