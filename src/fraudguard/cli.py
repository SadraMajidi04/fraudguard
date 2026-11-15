"""Command line entry point: ``python -m fraudguard.cli <command>``.

Commands: generate (data), train (experiment + model file), serve (HTTP API),
drift (compare new data with the training distribution).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from fraudguard.config import RANDOM_SEED
from fraudguard.data import (
    generate_transactions,
    load_transactions,
    save_transactions,
)
from fraudguard.features import build_features
from fraudguard.monitoring import drift_report
from fraudguard.pipeline import (
    ExperimentConfig,
    format_report,
    load_bundle,
    run_experiment,
    save_bundle,
)
from fraudguard.serving import OnlineScorer, make_server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraudguard", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a synthetic transactions file")
    gen.add_argument("--n", type=int, default=100_000, help="number of transactions")
    gen.add_argument("--fraud-rate", type=float, default=0.01)
    gen.add_argument("--seed", type=int, default=RANDOM_SEED)
    gen.add_argument("--out", type=Path, required=True, help="*.csv or *.csv.gz")

    tr = sub.add_parser("train", help="run the full experiment and save the best model")
    tr.add_argument("--data", type=Path, help="transactions file; omit to generate synthetic data")
    tr.add_argument("--n", type=int, default=100_000, help="rows to generate if --data is omitted")
    tr.add_argument("--fraud-rate", type=float, default=0.01)
    tr.add_argument("--seed", type=int, default=RANDOM_SEED)
    tr.add_argument("--review-cost", type=float, default=5.0)
    tr.add_argument("--out-dir", type=Path, default=Path("models"))

    sv = sub.add_parser("serve", help="start the scoring API")
    sv.add_argument("--model", type=Path, required=True)
    sv.add_argument("--warm-data", type=Path, help="past transactions to preload customer history")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)

    dr = sub.add_parser("drift", help="PSI drift report of new data vs the training data")
    dr.add_argument("--model", type=Path, required=True)
    dr.add_argument("--data", type=Path, required=True)
    return parser


def _cmd_generate(args) -> int:
    df = generate_transactions(n_transactions=args.n, fraud_rate=args.fraud_rate, seed=args.seed)
    path = save_transactions(df, args.out)
    n_fraud = int(df["is_fraud"].sum())
    print(f"wrote {len(df):,} rows ({n_fraud:,} fraud, {n_fraud / len(df):.2%}) to {path}")
    return 0


def _cmd_train(args) -> int:
    from fraudguard.evaluation import CostModel

    if args.data:
        df = load_transactions(args.data)
    else:
        df = generate_transactions(args.n, args.fraud_rate, seed=args.seed)
    cfg = ExperimentConfig(seed=args.seed, cost_model=CostModel(review_cost=args.review_cost))
    result = run_experiment(df, cfg)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    save_bundle(result.bundle, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(result.metrics, indent=2, allow_nan=False))
    report = format_report(result.metrics)
    (out / "report.md").write_text(report + "\n")
    print(report)
    print(f"\nselected model: {result.metrics['best_model']}  ->  {out / 'model.joblib'}")
    return 0


def _cmd_serve(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scorer = OnlineScorer(load_bundle(args.model))
    if args.warm_data:
        n = scorer.warm_start(load_transactions(args.warm_data))
        print(f"warm-started with {n:,} historical transactions")
    server = make_server(scorer, args.host, args.port)
    print(f"serving on http://{args.host}:{server.server_address[1]}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


def _cmd_drift(args) -> int:
    bundle = load_bundle(args.model)
    current = build_features(load_transactions(args.data))
    rep = drift_report(bundle["reference_features"], current)
    print(rep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    major = int((rep["status"] == "major").sum())
    print(f"\n{major} feature(s) with MAJOR drift (PSI >= 0.25)")
    return 3 if major else 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {"generate": _cmd_generate, "train": _cmd_train, "serve": _cmd_serve,
                "drift": _cmd_drift}
    try:
        return handlers[args.command](args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
