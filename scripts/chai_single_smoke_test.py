from pathlib import Path

import torch
from chai_lab.chai1 import run_inference


def main():
    protein_name = "rocklin_batch2_112316__top1"
    sequence = "ATETTEFTVKEGESPKKIGERLEKEGILPKNSGFAQELIDTGKAKTIEAKTYSIAKDATKQEIIDLISK"

    fasta_path = Path(f"/tmp/{protein_name}.fasta")
    fasta_path.write_text(f">protein|name={protein_name}\n{sequence}\n")

    output_dir = Path("/tmp/chai_single_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        seed=42,
        device=torch.device("cuda:0"),
        use_esm_embeddings=True,
    )

    print(candidates.cif_paths)


if __name__ == "__main__":
    main()
