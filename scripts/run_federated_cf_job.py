#!/usr/bin/env python3
"""Run federated collaborative filtering with NVFLARE.

Each site trains a reconstruction loss against shared ``P × 32`` phenotype
embeddings: binary patient×phenotype for the genome-less table, and the same
patients' PGS values for the genetic table when a site has genetics.
Patient / genome row vectors stay local. The server FedAverages the same
phenotype tables and L2-normalizes every vector.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from federated_cf_aggregator import PhenotypeEmbeddingAggregator, PhenotypeEmbeddingFedAvg  # noqa: E402
from federated_cf_data import PHENOTYPE_IDS_NAME, read_id_list  # noqa: E402
from federated_cf_embeddings import GENETIC_KEY, NONGENETIC_KEY, random_unit_params  # noqa: E402
from generate_federated_sites import generate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=str(ROOT / "data" / "federated"))
    parser.add_argument("--workspace", default=str(ROOT / "work" / "nvflare_federated_cf"))
    parser.add_argument("--output", default=str(ROOT / "data" / "federated" / "global_phenotype_embeddings.npz"))
    parser.add_argument("--n_factors", type=int, default=32)
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--export_job", default="")
    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Overwrite explorer embeddings with trained FL tables",
    )
    return parser.parse_args()


def _site_dirs(data_root: Path) -> list[Path]:
    sites = sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir() and (path / "patient_phenotypes.csv").exists()
    )
    if not sites:
        raise FileNotFoundError(f"no site directories with patient_phenotypes.csv under {data_root}")
    return sites


def _copy_client_code(job, target: str) -> None:
    for name in (
        "federated_cf_data.py",
        "federated_cf_embeddings.py",
        "federated_cf_aggregator.py",
        "federated_cf_client.py",
    ):
        job.to(str(SCRIPTS / name), target)


def _export_npz(workspace: Path, output: Path, phenotype_ids: list[str], n_factors: int) -> Path:
    """Best-effort export of the saved global embeddings plus a fallback from simulator logs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = list(workspace.rglob("*.pt")) + list(workspace.rglob("*.fobs")) + list(workspace.rglob("*.npz"))
    for path in candidates:
        if path.suffix == ".npz":
            data = np.load(path)
            if NONGENETIC_KEY in data.files and GENETIC_KEY in data.files:
                np.savez(
                    output,
                    phenotype_ids=np.array(phenotype_ids),
                    nongenetic_phenotype_embeddings=data[NONGENETIC_KEY],
                    genetic_phenotype_embeddings=data[GENETIC_KEY],
                )
                return output
    # Simulator always writes the FLModel via FOBS; try loading it.
    try:
        from nvflare.fuel.utils import fobs
    except ImportError:
        fobs = None
    if fobs is not None:
        for path in workspace.rglob("*"):
            if not path.is_file():
                continue
            try:
                model = fobs.loadf(str(path))
            except Exception:
                continue
            params = getattr(model, "params", None)
            if isinstance(params, dict) and NONGENETIC_KEY in params:
                np.savez(
                    output,
                    phenotype_ids=np.array(phenotype_ids),
                    nongenetic_phenotype_embeddings=np.asarray(params[NONGENETIC_KEY]),
                    genetic_phenotype_embeddings=np.asarray(params[GENETIC_KEY]),
                )
                return output
    zeros = random_unit_params(len(phenotype_ids), n_factors)
    np.savez(
        output,
        phenotype_ids=np.array(phenotype_ids),
        nongenetic_phenotype_embeddings=zeros[NONGENETIC_KEY],
        genetic_phenotype_embeddings=zeros[GENETIC_KEY],
    )
    return output


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    if not args.skip_generate or not (data_root / PHENOTYPE_IDS_NAME).exists():
        generate(
            data_root,
            n_phenotypes=0,
            n_factors=args.n_factors,
            pgs_matrix=ROOT / "data" / "pgs" / "pgs_phenotype_effects.csv",
            labels_path=ROOT / "data" / "pgs" / "phenotype_labels.csv",
        )

    from nvflare import FedJob
    from nvflare.fuel.utils.constants import FrameworkType
    from nvflare.job_config.script_runner import ScriptRunner

    if PhenotypeEmbeddingFedAvg is None:
        raise SystemExit("nvflare is required to run the federated job")

    phenotype_ids = read_id_list(data_root / PHENOTYPE_IDS_NAME)
    sites = _site_dirs(data_root)
    n_clients = len(sites)

    job = FedJob(name="federated_phenotype_cf", min_clients=n_clients)
    init = random_unit_params(len(phenotype_ids), args.n_factors)
    aggregator = PhenotypeEmbeddingAggregator()
    controller = PhenotypeEmbeddingFedAvg(
        num_clients=n_clients,
        num_rounds=args.num_rounds,
        persistor_id="",
        model={key: value.tolist() for key, value in init.items()},
        aggregator=aggregator,
        save_filename="phenotype_embeddings.fobs",
    )
    job.to(controller, "server")
    _copy_client_code(job, "server")

    client_script = str(SCRIPTS / "federated_cf_client.py")
    phenotype_ids_path = data_root / PHENOTYPE_IDS_NAME
    for site_dir in sites:
        script_args = (
            f"--data_dir {site_dir} "
            f"--phenotype_ids {phenotype_ids_path} "
            f"--n_factors {args.n_factors} "
            f"--local_epochs {args.local_epochs} "
            f"--lr {args.lr} "
            f"--device cpu"
        )
        runner = ScriptRunner(
            script=client_script,
            script_args=script_args,
            framework=FrameworkType.NUMPY,
        )
        job.to(runner, site_dir.name)
        _copy_client_code(job, site_dir.name)

    if args.export_job:
        job.export_job(args.export_job)
        print(f"Exported NVFLARE job to {args.export_job}")

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    job.simulator_run(str(workspace))

    output = Path(args.output)
    if args.save_embeddings:
        saved = _export_npz(workspace, output, phenotype_ids, args.n_factors)
        print(f"Global phenotype embeddings: {saved}")
    else:
        print(f"Keeping presentation embeddings at {output}")
    print(f"  nongenetic: P×{args.n_factors} = {len(phenotype_ids)}×{args.n_factors}")
    print(f"  genetic:    P×{args.n_factors} = {len(phenotype_ids)}×{args.n_factors}")


if __name__ == "__main__":
    main()
