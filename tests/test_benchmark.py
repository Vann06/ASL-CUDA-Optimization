"""Smoke tests reales del protocolo de benchmark, limitados a pocos batches."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch
from torch import nn

from src.data.asl_dataset import ASLNpyDataset, CNN2Augmentation
from src.models.cnn2 import create_cnn2, load_initial_checkpoint, make_adam
from src.train import TrainConfig, make_loader, run_repetition, train_one_epoch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT = ROOT / "results" / "initialization" / "cnn2_seed42_initial.pt"


class TestBenchmarkSmoke(unittest.TestCase):
    def test_one_real_batch_records_exclusive_timings(self) -> None:
        device = torch.device("cpu")
        dataset = ASLNpyDataset(DATA_DIR, "train")
        loader = make_loader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            seed=42,
            device=device,
        )
        model = create_cnn2(seed=42)
        load_initial_checkpoint(model, CHECKPOINT)
        result = train_one_epoch(
            model,
            CNN2Augmentation(),
            loader,
            make_adam(model),
            nn.CrossEntropyLoss(),
            device,
            max_batches=1,
        )
        self.assertEqual(result["examples"], 32)
        self.assertEqual(result["batches"], 1)
        self.assertTrue(math.isfinite(result["loss"]))
        for key in (
            "load_cpu_s",
            "transfer_s",
            "augmentation_s",
            "forward_s",
            "backward_s",
            "optimizer_s",
            "epoch_wall_s",
        ):
            self.assertGreaterEqual(result["timings"][key], 0.0)

    def test_reduced_epoch_restores_checkpoint_after_warmup(self) -> None:
        config = TrainConfig(
            device="cpu",
            epochs=1,
            batch_size=32,
            num_workers=0,
            seed=42,
            warmup_batches=1,
            max_train_batches=2,
            max_eval_batches=1,
        )
        result = run_repetition(
            repetition=1,
            config=config,
            data_dir=DATA_DIR,
            checkpoint_path=CHECKPOINT,
        )
        self.assertEqual(len(result["epochs"]), 1)
        self.assertEqual(result["epochs"][0]["train"]["examples"], 64)
        self.assertEqual(result["epochs"][0]["validation"]["examples"], 32)
        self.assertEqual(result["test"]["examples"], 32)
        self.assertEqual(result["checkpoint_metadata"]["seed"], 42)
        self.assertGreater(result["warmup_s_excluded"], 0.0)
        self.assertGreater(result["measured_execution_s"], 0.0)
        self.assertTrue(math.isfinite(result["test"]["f1_macro"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
