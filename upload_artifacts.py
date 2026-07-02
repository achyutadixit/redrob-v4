"""
upload_artifacts.py
-------------------
One-time script to upload precomputed artifacts to a Hugging Face Hub dataset repo.
This is run ONCE by the repository owner. Evaluators only need to run rank.py.

Usage:
  python upload_artifacts.py --repo YOUR_HF_USERNAME/redrob-v4-artifacts --token YOUR_HF_TOKEN
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo

ARTIFACTS = [
    "precompute/artifacts/embeddings.npy",
    "precompute/artifacts/jd_embedding.npy",
    "precompute/artifacts/bm25_scores.pkl",
    "precompute/artifacts/feature_matrix.pkl",
    "precompute/artifacts/honeypots_flagged.json",
]

def main():
    parser = argparse.ArgumentParser(description="Upload precomputed artifacts to Hugging Face Hub")
    parser.add_argument('--repo', type=str, required=True, help='HF repo id, e.g. username/redrob-v4-artifacts')
    parser.add_argument('--token', type=str, required=True, help='HF write access token')
    args = parser.parse_args()

    api = HfApi(token=args.token)

    # Create the dataset repo if it doesn't exist
    create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True, token=args.token)
    print(f"Uploading to: https://huggingface.co/datasets/{args.repo}")

    for artifact_path in ARTIFACTS:
        if not os.path.exists(artifact_path):
            print(f"  [SKIP] {artifact_path} not found.")
            continue
        filename = os.path.basename(artifact_path)
        size_mb = os.path.getsize(artifact_path) / (1024 * 1024)
        print(f"  Uploading {filename} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=artifact_path,
            path_in_repo=filename,
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(f"  Done: {filename}")

    print("\nAll artifacts uploaded. Set HF_REPO in rank.py or pass --hf-repo to use them.")

if __name__ == '__main__':
    main()
