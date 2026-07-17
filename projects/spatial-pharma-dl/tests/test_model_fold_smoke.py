"""Bounded offline smoke evidence for model tensors and LOSO orchestration."""

from __future__ import annotations

import pytest
import torch

import src.models as model_module
import src.train as train_module

pytestmark = pytest.mark.offline


class _TinyMultiHeadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.classifier = torch.nn.Linear(4, 2)
        self.regressor = torch.nn.Linear(4, 2)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(images)
        return self.classifier(encoded), self.regressor(encoded)


def _run_tiny_optimization(seed: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run a repeatable step without changing the caller's global RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _TinyMultiHeadModel().cpu()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        images = torch.rand((4, 3, 12, 12))
        classes = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        targets = torch.rand((4, 2))
        before = [parameter.detach().clone() for parameter in model.parameters()]

        optimizer.zero_grad()
        class_logits, regression = model(images)
        loss = torch.nn.functional.cross_entropy(
            class_logits, classes
        ) + torch.nn.functional.mse_loss(regression, targets)
        loss.backward()
        optimizer.step()

        assert class_logits.shape == (4, 2)
        assert regression.shape == (4, 2)
        assert torch.isfinite(loss)
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        assert any(
            not torch.equal(old, new.detach())
            for old, new in zip(before, model.parameters(), strict=True)
        )
        return loss.detach().clone(), [
            parameter.detach().clone() for parameter in model.parameters()
        ]


def test_tiny_multi_head_model_step_is_seeded_and_rng_isolated() -> None:
    global_state = torch.random.get_rng_state().clone()
    first_loss, first_parameters = _run_tiny_optimization(seed=41)
    torch.testing.assert_close(torch.random.get_rng_state(), global_state)
    second_loss, second_parameters = _run_tiny_optimization(seed=41)

    torch.testing.assert_close(second_loss, first_loss, rtol=0, atol=0)
    for first, second in zip(first_parameters, second_parameters, strict=True):
        torch.testing.assert_close(second, first, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), global_state)


def test_public_resnet18_shape_smoke_never_requests_weights(monkeypatch) -> None:
    original_weights = model_module._imagenet_weights

    def offline_weights(backbone: str, pretrained: bool):
        if pretrained:
            raise AssertionError("offline model smoke attempted pretrained weights")
        return original_weights(backbone, pretrained)

    monkeypatch.setattr(model_module, "_imagenet_weights", offline_weights)
    model = model_module.build_model(
        n_classes=3,
        n_genes=2,
        model_name="resnet18",
        pretrained=False,
    ).cpu()
    model.eval()

    with torch.inference_mode():
        class_logits, regression = model(torch.zeros((2, 3, 64, 64)))

    assert class_logits.shape == (2, 3)
    assert regression.shape == (2, 2)
    assert model.pretrained is False


def test_loso_folds_and_stubbed_orchestration_cover_each_slide_once(
    cohort_factory, monkeypatch, tmp_path
) -> None:
    cohort = cohort_factory()
    slide_ids = cohort["slide_ids"]
    labels = cohort["labels"]
    expected_folds = [
        (["slide_b", "slide_c"], "slide_a"),
        (["slide_a", "slide_c"], "slide_b"),
        (["slide_a", "slide_b"], "slide_c"),
    ]
    calls: list[tuple[int, tuple[str, ...], str]] = []

    def forbidden_output_path():
        raise AssertionError("stubbed LOSO smoke reached repository output setup")

    def train_one_fold_stub(
        train_slides,
        val_slide,
        _labels,
        cfg=None,
        fold=0,
        **_kwargs,
    ):
        assert cfg == {"seed": 17}
        calls.append((fold, tuple(train_slides), val_slide))
        return {
            "fold": fold,
            "train_slides": list(train_slides),
            "val_slide": val_slide,
        }

    monkeypatch.setattr(train_module, "pharma_outputs_dir", forbidden_output_path)
    monkeypatch.setattr(train_module, "train_one_fold", train_one_fold_stub)

    assert train_module.loso_folds(slide_ids) == expected_folds
    results = train_module.train_loso(slide_ids, labels, cfg={"seed": 17})

    assert [result["val_slide"] for result in results] == slide_ids
    assert calls == [
        (index, tuple(train_slides), held_out)
        for index, (train_slides, held_out) in enumerate(expected_folds)
    ]
    assert all(
        held_out not in train_slides for _, train_slides, held_out in calls
    )
    assert list(tmp_path.iterdir()) == []
