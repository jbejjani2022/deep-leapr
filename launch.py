#!/usr/bin/env python3
# Simple utility to launch jobs from the Makefile.

import os
import json
import argparse, shlex, subprocess, sys
from pathlib import Path


KNOWN_DOMAINS = ["chess", "image_classification", "text_classification", "text_regression", "pairwise_text_classification"]


def parse_domain_dataset(raw: str) -> tuple[str, str | None]:
    "If just the domain, return (domain, None). Otherwise if startswith <domain>_, split once."
    for dom in KNOWN_DOMAINS:
        if raw == dom:
            return dom, None
        prefix = f"{dom}_"
        if raw.startswith(prefix):
            return dom, raw[len(prefix) :]
    raise ValueError(
        f"{raw} did not match any of the supported domains: {KNOWN_DOMAINS}"
    )


def expand_model_name(name: str) -> str:
    "Tries to guess and add a provider prefix."

    if "/" in name:
        return name

    if name.startswith("gpt-"):
        return f"openai/{name}"
    if name.startswith("claude-"):
        return f"anthropic/{name}"

    return name


def ensure_dirs() -> None:
    Path("results/features").mkdir(parents=True, exist_ok=True)
    Path("results/evals").mkdir(parents=True, exist_ok=True)
    Path("results/shap").mkdir(parents=True, exist_ok=True)


def format_basename(method: str, domain_dataset: str, model: str) -> str:
    return f"{method}__{domain_dataset}__{model}"


def run(cmd: list[str], dry_run: bool) -> int:
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(printable)
    if dry_run:
        return 0
    return subprocess.run(cmd).returncode


def launch_leapr(learner: str, domain_raw: str, model: str, dry_run: bool) -> int:
    learner_arg = {"did3": "did3", "funsearch": "f2"}[learner]
    domain, dataset = parse_domain_dataset(domain_raw)
    model_full = expand_model_name(model)
    base = format_basename(learner, domain_raw, model)

    cmd = [
        "python",
        "main.py",
        f"learner={learner_arg}",
        f"domain={domain}",
        f"learner.model={model_full}",
        f"+output={base}",
    ]
    if domain in ("image_classification", "text_classification", "text_regression", "pairwise_text_classification"):
        if not dataset:
            print(
                f"error: domain '{domain}' requires a dataset suffix in '--domain'",
                file=sys.stderr,
            )
            return 2
        cmd.insert(4, f"dataset={dataset}")
    elif domain == "chess":
        # NOTE: could do cmd.insert(3, "dataset=lichess-eval") for uniformity
        pass

    return run(cmd, dry_run)


def launch_raw(domain_raw: str, dry_run: bool) -> int:
    domain, dataset = parse_domain_dataset(domain_raw)

    cmd = [
        "python",
        "main.py",
        "learner=raw",
        f"domain={domain}",
        f"dataset={dataset}",
        f"+output=raw__{domain_raw}",
    ]
    return run(cmd, dry_run)


def launch_train(learner: str, domain_raw: str, model: str, dry_run: bool) -> int:
    domain, dataset = parse_domain_dataset(domain_raw)
    base = format_basename(learner, domain_raw, model)
    features_path = Path("results/features") / f"{base}.json"

    if not dry_run and not features_path.exists():
        print(
            f"error: expected features file not found: {features_path}", file=sys.stderr
        )
        return 3

    if domain == "chess":
        cmd = [
            "python",
            "train.py",
            "domain=chess",
            "dataset=data/lichess-eval.jsonl",
            "trainer=random_forest",
            f"trainer.features_spec.file={features_path}",
        ]
    elif domain in ("image_classification", "text_classification", "text_regression", "pairwise_text_classification"):
        if not dataset:
            print(
                f"error: domain '{domain}' requires a dataset suffix in '--domain'",
                file=sys.stderr,
            )
            return 2
        
        # Determine task type based on domain
        if domain == "text_regression":
            task_type = "regression"
        else:
            task_type = "classification"
        
        cmd = [
            "python",
            "train.py",
            f"domain={domain}",
            f"+trainer.domain_name={domain}",
            f"+trainer.task_type={task_type}",
            f"dataset={dataset}",
            "trainer=random_forest",
            f"trainer.features_spec.file={features_path}",
        ]
    else:
        print(
            f"error: unknown domain '{domain}' parsed from --domain {domain_raw}",
            file=sys.stderr,
        )
        return 2

    return run(cmd, dry_run)


def combine_features(domain, model) -> None:
    features_dir = Path("results/features")
    output_file = Path(f"results/features/combo__{domain}__{model}.json")

    pattern = f"{domain}__{model}.json"

    matching_files = [
        f for f in features_dir.glob("*.json") if f.name.endswith(pattern)
    ]

    if not matching_files:
        print(f"No matching files found for pattern '{pattern}'.")
        return

    combined_features = []
    for file in matching_files:
        with open(file, "r") as f:
            data = json.load(f)
            features = data["used_features"] if isinstance(data, dict) else data
            combined_features.extend(features)

    combined_features = list(set(combined_features))

    output = {
        "combined_from": [str(f) for f in matching_files],
        "used_features": combined_features,
    }

    with open(output_file, "w") as out_f:
        json.dump(output, out_f, indent=2)

    print(f"Combined features from {len(matching_files)} files into {output_file}")


def launch_interpret(learner: str, domain_raw: str, model: str, dry_run: bool) -> int:
    """Launch SHAP interpretation analysis."""
    from interpret import generate_shap_report
    
    domain, dataset = parse_domain_dataset(domain_raw)
    base = format_basename(learner, domain_raw, model)
    model_path = Path("results/models") / f"{base}.pkl"
    
    if not model_path.exists():
        print(
            f"error: Model checkpoint not found: {model_path}\n"
            f"This model has not been trained yet. Please run --train first.",
            file=sys.stderr
        )
        return 4
    
    if dry_run:
        print(f"Would run SHAP interpretation on {model_path}")
        return 0
    
    try:
        report_path = generate_shap_report(learner, domain_raw, model)
        print(f"SHAP interpretation report saved to {report_path}")
        return 0
    except Exception as e:
        print(f"error: SHAP interpretation failed: {e}", file=sys.stderr)
        return 5


def main() -> int:
    p = argparse.ArgumentParser(description="Simpel launcher for LeaPR jobs.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--leapr", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--combine", action="store_true")
    mode.add_argument("--raw", action="store_true")
    mode.add_argument("--interpret", action="store_true")
    p.add_argument("--learner", required=False, choices=["did3", "funsearch", "combo"])
    p.add_argument(
        "--domain", required=True, help="e.g. chess or image_classification_mnist"
    )
    p.add_argument("--model", required=False)
    p.add_argument("--output", type=str)
    p.add_argument("-n", "--dry-run", action="store_true")
    args = p.parse_args()

    ensure_dirs()
    if args.leapr:
        assert args.learner in ("did3", "funsearch"), "Learner does not exist"
        assert args.model
        return launch_leapr(args.learner, args.domain, args.model, args.dry_run)
    elif args.combine:
        assert args.model
        return combine_features(args.domain, args.model)
    elif args.raw:
        return launch_raw(args.domain, args.dry_run)
    elif args.interpret:
        assert args.learner in ("did3", "funsearch", "combo"), "Learner must be specified"
        assert args.model, "Model must be specified"
        return launch_interpret(args.learner, args.domain, args.model, args.dry_run)
    else:
        return launch_train(args.learner, args.domain, args.model, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
