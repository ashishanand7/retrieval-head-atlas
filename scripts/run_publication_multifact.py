#!/usr/bin/env python
"""Run the frozen four-fact prose/query-selection study on SageMaker."""

import argparse
import gzip
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
from rha.multifact import FAMILIES, build_examples, condition_specs, extra_metrics, prior_ids, select_groups
from rha.multifact_scoring import assert_control
from scripts.run_publication_confirmation import tokenizer_fingerprint, write_json
from scripts.run_publication_specificity import source_hashes as specificity_source_hashes
from scripts.run_publication_temporal import verify_committed_sources

DELTA_FIELDS = ("clean_mean_logprob", "answer_logprob_margin", "clean_best_other_margin", "exact_clean",
                "clean_number_content", "first_number_is_clean", "clean_beats_all_candidates")
DELTA_NAMES = ("delta_clean_logprob", "delta_answer_margin", "delta_best_other_margin", "delta_exact_clean",
               "delta_clean_number_content", "delta_first_number_is_clean", "delta_clean_beats_all_candidates")


def source_hashes(protocol_path, original_path):
    hashes = specificity_source_hashes(protocol_path, original_path)
    for relative in ("rha/multifact.py", "rha/multifact_scoring.py", "scripts/run_publication_multifact.py",
                     "scripts/analyze_publication_multifact.py"):
        hashes[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    return hashes


def check_smoke(directory, protocol, hashes, packages, chunk_size):
    from scripts.analyze_publication_multifact import analyze
    if directory is None:
        raise ValueError("Confirmation requires --smoke-dir")
    result = analyze(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["arguments"]["stage"] != "smoke":
        raise ValueError("Smoke gate requires a smoke-stage run")
    for key, value in (("protocol", protocol), ("source_sha256", hashes), ("packages", packages)):
        if manifest[key] != value:
            raise ValueError(f"Smoke is stale: {key}; run a new smoke in an empty directory")
    if manifest["arguments"]["chunk_size"] != chunk_size or manifest["model_revision"] != protocol["revision"]:
        raise ValueError("Smoke model revision or chunk size differs")
    return {"directory": str(directory.resolve()), "rows_sha256": manifest["rows_sha256"],
            "references_sha256": manifest["references_sha256"], "validation": result["validation"]}


def iter_case_rows(model, tokenizer, layout, example, reference, groups, protocol, chunk_size):
    from rha.multifact_scoring import capture_reference, evaluate, prepare_prompt
    answers = example["candidate_answers"]
    prepared = {label: prepare_prompt(model, tokenizer, example[f"{label}_prompt"], answers,
                                     protocol["max_len"], chunk_size, protocol["max_new_tokens"])
                for label in ("clean", "corrupt")}
    if any(item.candidate_ids != example["candidate_token_ids"] or item.length != example["prompt_tokens"]
           or int(item.last_token.item()) != example["final_prompt_token_id"] for item in prepared.values()):
        raise RuntimeError("Prepared prompts/candidates disagree with saved examples")
    specs = condition_specs(example, groups, protocol)
    replacement = None
    if any(spec["action"] == "resample" for spec in specs.values()):
        if reference is None or reference["reference_id"] != example["reference_id"]:
            raise ValueError("Missing matched resampling reference")
        replacement = capture_reference(model, tokenizer, layout, reference, protocol["max_len"], chunk_size)
    baselines = {}
    metadata = {key: value for key, value in example.items() if key not in ("clean_prompt", "corrupt_prompt")}
    names = ["clean", "corrupt"] + [name for name in specs if name not in ("clean", "corrupt")]
    for name in names:
        spec = specs[name]
        values = evaluate(model, tokenizer, layout, prepared[spec["recipient"]], protocol["max_new_tokens"],
                          action=spec["action"], pairs=spec["pairs"], scope=spec["scope"],
                          donor=prepared.get(spec["donor"]), resample=replacement if spec["action"] == "resample" else None)
        values.update({f"{label}_mean_logprob": score["mean_logprob"] for label, score in values["candidate_scores"].items()})
        values.update(extra_metrics(values["generated_text"], values["candidate_scores"], answers, example["resample_gold"]))
        if name in ("clean", "corrupt"):
            baselines[name] = values
        baseline = baselines[spec["recipient"]]
        yield {**metadata, "condition": name, "reference": spec["recipient"], **values,
               **{delta: values[field] - baseline[field] for field, delta in zip(DELTA_FIELDS, DELTA_NAMES)}}
        if name.startswith("self_patch_"):
            assert_control(values, baseline)
        if name == "positive_transfer":
            assert_control(values, baselines["clean"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--protocol", default="configs/publication_multifact.json")
    parser.add_argument("--stage", choices=("smoke", "confirmation"), default="smoke")
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol_path = resolve_repo_path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    original_path = resolve_repo_path(protocol["confirmation_protocol"])
    original = json.loads(original_path.read_text())
    groups = select_groups(protocol, original)
    cfg = load_runtime_config(str(resolve_repo_path(args.config)))
    if cfg.model_id != protocol["model_id"] or cfg.precision.lower() not in ("auto", "bf16", "bfloat16"):
        raise ValueError("Frozen multi-fact study requires the configured Qwen model in bfloat16")
    if args.chunk_size < 1 or protocol["max_new_tokens"] < 1:
        raise ValueError("Invalid inference budget")
    if args.stage == "confirmation" and not args.prepare_only and args.smoke_dir is None:
        raise ValueError("Confirmation requires --smoke-dir")
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
            raise RuntimeError("Run pretrained experiments on SageMaker; locally use --prepare-only or tests")
        verify_committed_sources(hashes, revision)
        if args.stage == "confirmation":
            smoke = check_smoke(args.smoke_dir, protocol, hashes, packages, args.chunk_size)
    settings = protocol["stages"][args.stage]
    out.mkdir(parents=True, exist_ok=True)
    count = settings["n_per_family"] * len(FAMILIES)
    manifest = {"status": "preparing", "protocol": protocol, "original_protocol": original, "groups": groups,
        "protocol_source": str(protocol_path.relative_to(ROOT)), "original_protocol_source": str(original_path.relative_to(ROOT)),
        "runtime_config": asdict(cfg), "source_sha256": hashes, "git_commit": revision,
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": platform.python_version(), "packages": packages, "smoke_gate": smoke,
        "n_per_family": settings["n_per_family"], "resolved_seed_base": settings["seed_base"],
        "n_independent_example_ids": count, "n_reference_identities": count, "rows_written": 0,
        "intervention_note": "No prefill intervention. Dynamic patches use the recipient prefix. Resampling uses a separate task-matched query state once, not an aligned decoding donor.",
        "positive_control_note": "Full pre-final-norm residual replacement must reproduce current donor logits, all four candidate scores and donor-baseline generation, even when the donor is wrong.",
        "candidate_note": "All four codes are plausible facts; the legacy scorer label decoy denotes another fact, not an explicitly irrelevant distractor.",
        "storage_note": "rows.jsonl.gz is lossless. rows_sha256 hashes uncompressed JSONL bytes; rows_gzip_sha256 hashes the archive."}
    write_json(out / "manifest.json", manifest)
    started = time.monotonic()
    try:
        from transformers import AutoTokenizer
        if cfg.hf_home:
            os.environ["HF_HOME"] = os.path.expanduser(cfg.hf_home)
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=protocol["revision"],
                                                 trust_remote_code=False, local_files_only=args.local_files_only)
        fingerprint = tokenizer_fingerprint(tokenizer)
        if fingerprint != protocol["tokenizer_and_chat_template_sha256"]:
            raise ValueError("Pinned tokenizer fingerprint differs")
        forbidden = prior_ids(ROOT, protocol["protocol"])
        examples, references = build_examples(tokenizer, protocol, args.stage, forbidden)
        for filename, records in (("examples", examples), ("references", references)):
            with (out / f"{filename}.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
            manifest[f"{filename}_sha256"] = hashlib.sha256((out / f"{filename}.jsonl").read_bytes()).hexdigest()
        manifest.update(status="prepared", n_example_pairs=len(examples), n_reference_prompts=len(references),
                        expected_rows=sum(len(condition_specs(e, groups, protocol)) for e in examples),
                        prior_identity_count_checked=len(forbidden), prior_identity_overlap=0,
                        tokenizer_revision_requested=protocol["revision"], tokenizer_and_chat_template_sha256=fingerprint)
        write_json(out / "manifest.json", manifest)
        print(f"Prepared {len(examples)} cases, {count} study identities, {len(references)} reference prompts, "
              f"{manifest['expected_rows']} planned rows: {out}", flush=True)
        if args.prepare_only:
            return
        from transformers import AutoModelForCausalLM
        from tqdm.auto import tqdm
        from rha.modeling import inspect_model_layout
        from rha.seed import set_seed
        set_seed(settings["seed_base"])
        model_started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, revision=protocol["revision"], torch_dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa", trust_remote_code=False, local_files_only=args.local_files_only).eval()
        layout = inspect_model_layout(model)
        if (len(layout.layers), layout.num_heads, model.config.model_type) != (protocol["num_layers"], protocol["num_heads"], "qwen2"):
            raise ValueError("Model layout differs from frozen multi-fact protocol")
        if model.config._commit_hash != protocol["revision"]:
            raise ValueError("Loaded model revision differs")
        eos = model.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        if not eos or any(not isinstance(token, int) for token in eos):
            raise ValueError("Missing valid EOS IDs")
        manifest.update(status="running", model_revision=model.config._commit_hash, device=torch.cuda.get_device_name(),
                        dtype=str(model.dtype), cuda_version=torch.version.cuda, eos_token_ids=eos)
        torch.cuda.reset_peak_memory_stats()
        write_json(out / "manifest.json", manifest)
        refs = {reference["reference_id"]: reference for reference in references}
        row_hash = hashlib.sha256()
        with gzip.open(out / "rows.jsonl.gz", "wb") as handle:
            for example in tqdm(examples, desc=f"multi-fact {args.stage}"):
                for row in iter_case_rows(model, tokenizer, layout, example, refs.get(example["reference_id"]), groups, protocol, args.chunk_size):
                    encoded = (json.dumps(row, allow_nan=False) + "\n").encode("utf-8")
                    handle.write(encoded)
                    handle.flush()
                    row_hash.update(encoded)
                    manifest["rows_written"] += 1
                write_json(out / "manifest.json", manifest)
        if source_hashes(protocol_path, original_path) != hashes:
            raise RuntimeError("Experiment sources changed during run; retain outputs for inspection")
        manifest.update(status="complete", elapsed_seconds_total=time.monotonic() - started,
                        elapsed_seconds_including_model_load=time.monotonic() - model_started,
                        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(), rows_sha256=row_hash.hexdigest(),
                        rows_gzip_sha256=hashlib.sha256((out / "rows.jsonl.gz").read_bytes()).hexdigest())
        write_json(out / "manifest.json", manifest)
        from scripts.analyze_publication_multifact import analyze, write_analysis
        result = analyze(out)
        write_analysis(out, result)
        print(f"PASS: {manifest['rows_written']} rows; four-candidate positive transfer, self-patch, pairing and trace audits passed.", flush=True)
        print(f"Baseline counts: {result['validation']['baseline_counts']}", flush=True)
        print(f"Complete scientific outputs: {out}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds_total=time.monotonic() - started)
        write_json(out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
