"""Pruebas de CNN2, dataset, augmentation y un paso Adam en CPU."""

from __future__ import annotations

import tempfile
import unittest
import pickle
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.asl_dataset import ASLNpyDataset, CNN2Augmentation
from src.models.cnn2 import (
    EXPECTED_KERAS_TOTAL,
    EXPECTED_TRAINABLE_PARAMETERS,
    create_cnn2,
    keras_compatible_parameter_count,
    load_initial_checkpoint,
    make_adam,
    save_initial_checkpoint,
    trainable_parameter_count,
)
from src.utils.device import select_device


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BATCH_SIZE = 32


class TestCNN2Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_dataset = ASLNpyDataset(DATA_DIR, "train")
        cls.val_dataset = ASLNpyDataset(DATA_DIR, "val")
        loader = DataLoader(
            cls.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        cls.images, cls.labels = next(iter(loader))

    def test_dataset_shapes_labels_and_range(self) -> None:
        self.assertEqual(len(self.train_dataset), 12_180)
        self.assertEqual(len(self.val_dataset), 2_610)
        self.assertEqual(tuple(self.images.shape), (32, 3, 64, 64))
        self.assertEqual(tuple(self.labels.shape), (32,))
        self.assertEqual(self.images.dtype, torch.float32)
        self.assertEqual(self.labels.dtype, torch.int64)
        self.assertGreaterEqual(float(self.images.min()), 0.0)
        self.assertLessEqual(float(self.images.max()), 1.0)
        self.assertGreaterEqual(int(self.labels.min()), 0)
        self.assertLess(int(self.labels.max()), 29)
        self.assertEqual(int(self.train_dataset.labels.min()), 0)
        self.assertEqual(int(self.train_dataset.labels.max()), 28)
        self.assertEqual(len(self.train_dataset.class_names), 29)

    def test_exact_layer_configuration(self) -> None:
        model = create_cnn2(seed=42)
        self.assertEqual(model.conv1.kernel_size, (3, 3))
        self.assertEqual(model.conv1.padding, (1, 1))
        self.assertEqual((model.conv1.in_channels, model.conv1.out_channels), (3, 32))
        self.assertEqual((model.conv2.in_channels, model.conv2.out_channels), (32, 64))
        self.assertEqual((model.conv3.in_channels, model.conv3.out_channels), (64, 128))
        self.assertEqual(model.pool1.kernel_size, 2)
        self.assertEqual(model.dropout1.p, 0.25)
        self.assertEqual(model.dropout2.p, 0.25)
        self.assertEqual(model.dropout3.p, 0.25)
        self.assertEqual(model.dropout4.p, 0.50)
        self.assertEqual(model.bn1.eps, 1e-3)
        self.assertEqual(model.bn1.momentum, 0.01)
        self.assertEqual((model.classifier.in_features, model.classifier.out_features), (128, 29))

    def test_dataset_is_deterministic_for_val_and_test_policy(self) -> None:
        first, label_a = self.val_dataset[0]
        second, label_b = self.val_dataset[0]
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(int(label_a), int(label_b))

    def test_dataset_reopens_memmap_in_windows_workers(self) -> None:
        serialized = pickle.dumps(self.train_dataset)
        self.assertLess(len(serialized), 20_000)
        restored = pickle.loads(serialized)
        image, label = restored[0]
        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertGreaterEqual(int(label), 0)

    def test_augmentation_shape_range_and_eval_identity(self) -> None:
        augmentation = CNN2Augmentation()
        augmentation.train()
        torch.manual_seed(42)
        augmented = augmentation(self.images.clone())
        self.assertEqual(augmented.shape, self.images.shape)
        self.assertTrue(bool(torch.isfinite(augmented).all()))
        self.assertGreaterEqual(float(augmented.min()), 0.0)
        self.assertLessEqual(float(augmented.max()), 1.0)
        self.assertFalse(torch.equal(augmented, self.images))

        augmentation.eval()
        unchanged = augmentation(self.images)
        self.assertTrue(torch.equal(unchanged, self.images))

    def test_model_shape_and_parameter_counts(self) -> None:
        model = create_cnn2(seed=42)
        self.assertEqual(trainable_parameter_count(model), EXPECTED_TRAINABLE_PARAMETERS)
        self.assertEqual(keras_compatible_parameter_count(model), EXPECTED_KERAS_TOTAL)
        model.eval()
        with torch.no_grad():
            logits = model(self.images)
        self.assertEqual(tuple(logits.shape), (32, 29))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_forward_backward_and_adam_update(self) -> None:
        model = create_cnn2(seed=42).to(select_device("cpu"))
        model.train()
        optimizer = make_adam(model)
        self.assertEqual(optimizer.defaults["lr"], 1e-3)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))
        self.assertEqual(optimizer.defaults["eps"], 1e-7)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.0)
        criterion = nn.CrossEntropyLoss()
        before = model.conv1.weight.detach().clone()

        augmentation = CNN2Augmentation().train()
        torch.manual_seed(42)
        training_images = augmentation(self.images.clone())

        optimizer.zero_grad(set_to_none=True)
        logits = model(training_images)
        loss = criterion(logits, self.labels)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertIsNotNone(model.conv1.weight.grad)
        self.assertTrue(bool(torch.isfinite(model.conv1.weight.grad).all()))
        optimizer.step()

        self.assertFalse(torch.equal(before, model.conv1.weight.detach()))

    def test_checkpoint_reproduces_initial_state(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "results") as temporary:
            checkpoint = Path(temporary) / "initial.pt"
            save_initial_checkpoint(checkpoint, seed=42)
            expected = create_cnn2(seed=42)
            restored = create_cnn2(seed=999)
            metadata = load_initial_checkpoint(restored, checkpoint)
            self.assertEqual(metadata["seed"], 42)
            self.assertEqual(metadata["keras_compatible_parameters"], 97_885)
            for expected_tensor, restored_tensor in zip(
                expected.state_dict().values(), restored.state_dict().values()
            ):
                self.assertTrue(torch.equal(expected_tensor, restored_tensor))

    def test_cuda_request_is_strict_when_unavailable(self) -> None:
        if torch.cuda.is_available():
            self.assertEqual(select_device("cuda").type, "cuda")
        else:
            with self.assertRaisesRegex(RuntimeError, "Se solicitó CUDA"):
                select_device("cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
