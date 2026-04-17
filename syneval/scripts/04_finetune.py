#!/usr/bin/env python3
"""SynEval Step 4: LoRA fine-tune a model on a filtered data subset.

Supports Qwen2.5-7B-Instruct and LLaMA-3-8B-Instruct.
Designed for 8×RTX 4090 with multi-GPU training via torchrun.

Correctly handles CUDA_VISIBLE_DEVICES for 2×4-GPU parallel training:
  - When CUDA_VISIBLE_DEVICES=4,5,6,7 and torchrun --nproc_per_node=4,
    local_rank 0-3 maps to the *visible* devices (indices 0-3 in the
    remapped space), NOT physical GPU IDs. We let the Trainer handle
    device placement — no manual model.to(cuda:X).

Usage (single run)::

    python syneval/scripts/04_finetune.py \\
        --dataset syneval/data/subsets/exp1_full.jsonl \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --output-dir syneval/results/models/exp1_full_qwen \\
        --config-name exp1_full --run-id 1

Usage (torchrun multi-GPU, all 8 GPUs)::

    torchrun --nproc_per_node=8 syneval/scripts/04_finetune.py \\
        --dataset syneval/data/subsets/exp1_full.jsonl \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --output-dir syneval/results/models/exp1_full_qwen_run1 \\
        --config-name exp1_full --run-id 1

Usage (torchrun 4 GPUs with CUDA_VISIBLE_DEVICES)::

    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \\
        syneval/scripts/04_finetune.py ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# LoRA hyperparameters from the paper plan
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training hyperparameters
# RTX 4090 (24GB): Qwen2.5-7B bf16 ≈ 14GB base, LoRA adds ~1GB.
# With gradient_checkpointing=True → ~18-20GB per GPU.
# Effective batch = batch_size × ngpu × grad_accum (auto-calculated by shell).
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4   # default; overridden by --gradient-accumulation
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 2048


# ---------------------------------------------------------------------------
# Chat format helpers
# ---------------------------------------------------------------------------
def _get_chat_template(model_name: str) -> str:
    """Return the appropriate chat template tag for the model."""
    name_lower = model_name.lower()
    if "llama" in name_lower:
        return "llama3"
    return "qwen"  # default for Qwen models


def format_as_chat(record: dict, model_name: str) -> str:
    """Format instruction-response as a chat string."""
    synth = record.get("synthetic_data") or record
    instruction = str(synth.get("instruction", "")).strip()
    response = str(synth.get("response", synth.get("output", ""))).strip()

    name_lower = model_name.lower()
    if "llama" in name_lower:
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{response}<|eot_id|>"
        )
    else:
        # Qwen / ChatML format
        return (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )


def load_training_data(dataset_path: str, model_name: str) -> list[str]:
    """Load JSONL and format as chat strings."""
    texts: list[str] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = format_as_chat(rec, model_name)
            texts.append(text)
    return texts


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
def run_finetune(args: argparse.Namespace) -> None:
    # Set BEFORE importing torch so the allocator picks it up
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    try:
        from trl import SFTTrainer, SFTConfig
        _HAS_SFT_CONFIG = True
    except ImportError:
        from trl import SFTTrainer
        _HAS_SFT_CONFIG = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main = local_rank <= 0

    if is_main:
        logger.info("Loading training data from %s", args.dataset)

    texts = load_training_data(args.dataset, args.base_model)
    if is_main:
        logger.info("Loaded %d training examples", len(texts))

    dataset = Dataset.from_dict({"text": texts})

    if is_main:
        logger.info("Loading model: %s", args.base_model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- FIX: correct device handling for CUDA_VISIBLE_DEVICES ----
    # When torchrun sets LOCAL_RANK, each process should load the model
    # directly onto its assigned GPU using device_map with the local_rank.
    # CUDA_VISIBLE_DEVICES remaps physical GPUs so local_rank 0-3 always
    # maps to visible indices 0-3 regardless of physical GPU IDs.
    if local_rank >= 0:
        # Load directly onto the correct GPU to avoid CPU→GPU copy OOM
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": local_rank},
            low_cpu_mem_usage=True,
        )
    else:
        # Single-GPU or CPU: let transformers pick the device
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    common_train_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        optim="adamw_torch_fused",
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    if _HAS_SFT_CONFIG:
        training_args = SFTConfig(
            **common_train_kwargs,
            max_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
        )
    else:
        training_args = TrainingArguments(**common_train_kwargs)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            peft_config=lora_config,
            tokenizer=tokenizer,
            max_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
        )

    if is_main:
        logger.info(
            "Training: epochs=%d batch_size=%d grad_accum=%d lora_rank=%d grad_ckpt=True",
            args.epochs, args.batch_size, args.gradient_accumulation, args.lora_rank,
        )

    trainer.train()

    # ---- FIX: only rank 0 saves, then verify the file exists ----
    final_path = output_dir / "final"
    if is_main:
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))

        # Verify adapter was actually written
        adapter_file = final_path / "adapter_model.safetensors"
        if adapter_file.exists():
            size_mb = adapter_file.stat().st_size / (1024 * 1024)
            logger.info("Model saved to %s (%.1f MB)", final_path, size_mb)
        else:
            logger.error(
                "SAVE FAILED: adapter_model.safetensors not found in %s",
                final_path,
            )
            sys.exit(1)

        meta = {
            "config_name": args.config_name,
            "run_id": args.run_id,
            "base_model": args.base_model,
            "dataset": args.dataset,
            "num_samples": len(texts),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "lora_rank": args.lora_rank,
            "output_dir": str(final_path),
        }
        meta_path = output_dir / "training_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Training metadata written to %s", meta_path)

    # Wait for rank 0 to finish saving before other ranks exit
    if local_rank >= 0:
        torch.distributed.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SynEval Step 4: LoRA fine-tune on filtered subset."
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to filtered subset JSONL.")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model name/path.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save fine-tuned model.")
    parser.add_argument("--config-name", type=str, required=True,
                        help="Filter config name (e.g. exp1_full).")
    parser.add_argument("--run-id", type=int, default=1,
                        help="Run ID for repeated training (1, 2, or 3).")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gradient-accumulation", type=int, default=GRADIENT_ACCUMULATION)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    run_finetune(args)


if __name__ == "__main__":
    main()
