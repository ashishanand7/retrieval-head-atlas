#!/usr/bin/env python
"""Run the frozen, prefix-aligned temporal patching study on SageMaker."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rha.config import load_runtime_config, resolve_repo_path
from rha.temporal import SCOPES, build_examples, condition_names, prior_identities, select_groups
from scripts.run_publication_confirmation import tokenizer_fingerprint, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--protocol", default="configs/publication_temporal.json")
    parser.add_argument("--stage", choices=("smoke", "confirmation"), default="smoke")
    parser.add_argument("--smoke-dir", type=Path, help="Required for GPU confirmation; must contain a passing smoke from identical sources")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--prepare-only", action="store_true", help="Validate and save prompts without loading model weights")
    parser.add_argument("--local-files-only", action="store_true", help="Use already cached pinned Hugging Face files")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def source_hashes(protocol_path, original_path):
    files = [Path(__file__), ROOT / "scripts/analyze_publication_temporal.py",
             ROOT / "scripts/analyze_publication_confirmation.py", ROOT / "scripts/run_publication_confirmation.py",
             *[ROOT / f"rha/{name}.py" for name in ("temporal", "temporal_scoring", "confirmation", "confirmation_scoring",
                                                  "prompts", "generation", "interventions", "modeling", "config", "seed")],
             protocol_path, original_path]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def verify_committed_sources(hashes, revision):
    for relative, expected in hashes.items():
        try:
            content = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=ROOT, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            raise ValueError("Commit experiment sources before running the GPU study; use --prepare-only while editing") from exc
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Uncommitted experiment source: {relative}; commit and sync before the GPU study")


def check_smoke(directory, protocol, hashes, packages, chunk_size):
    from scripts.analyze_publication_temporal import analyze
    if directory is None:
        raise ValueError("Confirmation requires --smoke-dir from a successful smoke run")
    result = analyze(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["arguments"]["stage"] != "smoke":
        raise ValueError("The smoke gate requires a smoke-stage run")
    for name, expected in (("protocol", protocol), ("source_sha256", hashes), ("packages", packages)):
        if manifest[name] != expected:
            raise ValueError(f"Smoke is stale or from a different environment: {name}; run a new smoke")
    if manifest["arguments"]["chunk_size"] != chunk_size or manifest["model_revision"] != protocol["revision"]:
        raise ValueError("Smoke uses different chunking or model revision")
    return {"directory": str(directory.resolve()), "rows_sha256": manifest["rows_sha256"], "validation": result["validation"]}


def assert_self_patch(values, baseline):
    if values["generated_token_ids"] != baseline["generated_token_ids"] or values["generation_stop"] != baseline["generation_stop"]:
        raise RuntimeError("Temporal self-patch changed generated tokens or termination")
    arrays = [(values["generated_token_logprobs"], baseline["generated_token_logprobs"])]
    arrays += [(values["candidate_scores"][key]["token_logprobs"], baseline["candidate_scores"][key]["token_logprobs"])
               for key in ("clean", "corrupt", "decoy")]
    if any(len(left) != len(right) or any(abs(a - b) > 1e-5 for a, b in zip(left, right)) for left, right in arrays):
        raise RuntimeError("Temporal self-patch changed per-token scores")


def iter_case_rows(model, tokenizer, layout, example, groups, protocol, chunk_size):
    from rha.temporal_scoring import evaluate, prepare_prompt
    answers = {"clean": example["clean_gold"], "corrupt": example["corrupt_gold"], "decoy": example["unrelated_decoy"]}
    prepared = {label: prepare_prompt(model, tokenizer, example[f"{label}_prompt"], answers,
                                     protocol["max_len"], chunk_size, protocol["max_new_tokens"])
                for label in ("clean", "corrupt")}
    if any(value.candidate_ids != example["candidate_token_ids"] for value in prepared.values()):
        raise RuntimeError("Scoring candidates differ from prepared examples")
    baselines = {label: evaluate(model, tokenizer, layout, value, protocol["max_new_tokens"])
                 for label, value in prepared.items()}
    metadata = {key: value for key, value in example.items() if key not in ("clean_prompt", "corrupt_prompt")}

    def row(condition, values, reference):
        baseline = baselines[reference]
        return {**metadata, "condition": condition, "reference": reference, **values,
                "delta_clean_logprob": values["clean_mean_logprob"] - baseline["clean_mean_logprob"],
                "delta_answer_margin": values["answer_logprob_margin"] - baseline["answer_logprob_margin"],
                "delta_clean_decoy_margin": values["clean_decoy_logprob_margin"] - baseline["clean_decoy_logprob_margin"],
                "delta_exact_clean": values["exact_clean"] - baseline["exact_clean"]}

    for label, values in baselines.items():
        yield row(label, values, label)
    selected = sorted({pair for pairs in groups.values() for pair in pairs})
    for label in ("clean", "corrupt"):
        for scope in SCOPES:
            values = evaluate(model, tokenizer, layout, prepared[label], protocol["max_new_tokens"],
                              donor=prepared[label], pairs=selected, scope=scope)
            yield row(f"self_patch_{label}:{scope}", values, label)
            assert_self_patch(values, baselines[label])
    for name, pairs in groups.items():
        for scope in SCOPES:
            values = evaluate(model, tokenizer, layout, prepared["corrupt"], protocol["max_new_tokens"],
                              donor=prepared["clean"], pairs=pairs, scope=scope)
            yield row(f"patch:{scope}:{name}", values, "corrupt")
            if scope == "decode_only":
                baseline = baselines["corrupt"]
                if values["generated_token_ids"][0] != baseline["generated_token_ids"][0] or any(
                    abs(values["candidate_scores"][label]["token_logprobs"][0] - baseline["candidate_scores"][label]["token_logprobs"][0]) > 1e-5
                    for label in answers
                ):
                    raise RuntimeError("Decoding-only patch changed the unpatched first prediction")


def main(argv=None):
    args = parse_args(argv)
    protocol_path = resolve_repo_path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    original_path = resolve_repo_path(protocol["confirmation_protocol"])
    original = json.loads(original_path.read_text())
    cfg = load_runtime_config(str(resolve_repo_path(args.config)))
    if cfg.model_id != protocol["model_id"] or cfg.precision.lower() not in ("auto", "bf16", "bfloat16"):
        raise ValueError("This frozen study requires the configured Qwen model in bfloat16")
    if tuple(protocol["scopes"]) != SCOPES or args.chunk_size < 1:
        raise ValueError("Unexpected scopes or invalid chunk size")
    if args.stage == "confirmation" and not args.prepare_only and args.smoke_dir is None:
        raise ValueError("Confirmation requires --smoke-dir from a successful smoke run")
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite or append to a run: {out}")
    hashes = source_hashes(protocol_path, original_path)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate", "numpy")}
    smoke = None
    if not args.prepare_only:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Run model experiments on SageMaker; locally use --prepare-only or unit tests")
        verify_committed_sources(hashes, revision)
        if args.stage == "confirmation":
            smoke = check_smoke(args.smoke_dir, protocol, hashes, packages, args.chunk_size)
    groups = select_groups(protocol, original)
    stage = protocol["stages"][args.stage]
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "preparing", "protocol": protocol, "original_protocol": original, "groups": groups,
        "runtime_config": asdict(cfg), "source_sha256": hashes, "git_commit": revision,
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": platform.python_version(), "packages": packages, "smoke_gate": smoke,
        "n_per_variant": stage["n_per_variant"], "resolved_seed_base": stage["seed_base"],
        "n_independent_example_ids": stage["n_per_variant"] * 5, "rows_written": 0,
        "intervention_note": "Unpatched donor and recipient consume the same prefix; caches isolated across conditions and branches. No prefill interventions.",
    }
    write_json(out / "manifest.json", manifest)
    total_start, model_start = time.monotonic(), None
    try:
        from transformers import AutoTokenizer
        if cfg.hf_home:
            os.environ["HF_HOME"] = os.path.expanduser(cfg.hf_home)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=protocol["revision"],
                                                 trust_remote_code=False, local_files_only=args.local_files_only)
        fingerprint = tokenizer_fingerprint(tokenizer)
        if fingerprint != protocol["tokenizer_and_chat_template_sha256"]:
            raise ValueError("Pinned tokenizer/chat-template fingerprint differs from confirmation")
        forbidden = prior_identities(ROOT, protocol["protocol"])
        examples = build_examples(tokenizer, protocol, args.stage, forbidden)
        with (out / "examples.jsonl").open("w") as handle:
            for example in examples:
                handle.write(json.dumps(example, allow_nan=False) + "\n")
        manifest.update({"status": "prepared", "n_example_pairs": len(examples),
                         "expected_rows": len(examples) * len(condition_names(groups)),
                         "examples_sha256": hashlib.sha256((out / "examples.jsonl").read_bytes()).hexdigest(),
                         "prior_identity_count_checked": len(forbidden), "prior_identity_overlap": 0,
                         "tokenizer_revision_requested": protocol["revision"],
                         "tokenizer_and_chat_template_sha256": fingerprint})
        write_json(out / "manifest.json", manifest)
        print(f"Prepared {len(examples)} cases, {manifest['n_independent_example_ids']} identities, "
              f"{manifest['expected_rows']} planned rows: {out}", flush=True)
        if args.prepare_only:
            return
        from transformers import AutoModelForCausalLM
        from tqdm.auto import tqdm
        from rha.modeling import inspect_model_layout
        from rha.seed import set_seed
        set_seed(stage["seed_base"])
        model_start = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, revision=protocol["revision"], torch_dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=False,
            local_files_only=args.local_files_only,
        ).eval()
        layout = inspect_model_layout(model)
        if (len(layout.layers), layout.num_heads) != (protocol["num_layers"], protocol["num_heads"]):
            raise ValueError("Model layout differs from frozen head configuration")
        if model.config._commit_hash != protocol["revision"]:
            raise ValueError("Loaded model revision differs from protocol")
        eos = model.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        if not eos or any(not isinstance(token, int) for token in eos):
            raise ValueError("Missing or invalid EOS token IDs")
        manifest.update({"status": "running", "device": torch.cuda.get_device_name(), "dtype": str(model.dtype),
                         "model_revision": model.config._commit_hash, "cuda_version": torch.version.cuda,
                         "eos_token_ids": eos})
        torch.cuda.reset_peak_memory_stats()
        write_json(out / "manifest.json", manifest)
        with (out / "rows.jsonl").open("w") as handle:
            for example in tqdm(examples, desc=f"temporal {args.stage}"):
                for row in iter_case_rows(model, tokenizer, layout, example, groups, protocol, args.chunk_size):
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
                    manifest["rows_written"] += 1
                write_json(out / "manifest.json", manifest)
        if source_hashes(protocol_path, original_path) != hashes:
            raise RuntimeError("Experiment source files changed during the run; retain outputs for inspection")
        manifest.update({"status": "complete", "elapsed_seconds_including_model_load": time.monotonic() - model_start,
                         "elapsed_seconds_total": time.monotonic() - total_start,
                         "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                         "rows_sha256": hashlib.sha256((out / "rows.jsonl").read_bytes()).hexdigest()})
        write_json(out / "manifest.json", manifest)
        from scripts.analyze_publication_temporal import analyze, write_analysis
        result = analyze(out)
        write_analysis(out, result)
        print(f"PASS: {manifest['rows_written']} rows; identity, prefix, hook, cache, and output audits passed.", flush=True)
        print(f"Baseline exact counts: {result['validation']['baseline_exact_counts']}", flush=True)
        print(f"Summary, paired analysis, validation, manifests and per-step outputs: {out}", flush=True)
    except BaseException as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                         "elapsed_seconds_total": time.monotonic() - total_start})
        write_json(out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
