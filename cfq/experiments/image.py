from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..data.image import load_image_dataset
from ..checkpoints import save_quantization_settings
from ..latent import LatentRecourseSolver
from ..models import ConvAutoencoder, QuantSmallCNN, SmallCNN, copy_fp_to_quantized
from ..quantization import bit_cost, hard_bit_allocation, iter_quant_layers
from ..training import magnitude_prune
from ..utils import resolve_device, save_json, seed_everything


def _loader(x, y, batch_size, shuffle, seed=42):
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def _train_classifier(model, bundle, device, epochs=5, batch_size=128, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = _loader(bundle.x_train, bundle.y_train, batch_size, True)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
    return model


def _train_autoencoder(autoencoder, bundle, device, epochs=5, batch_size=128, lr=1e-3):
    autoencoder.to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    loader = _loader(bundle.x_train, bundle.y_train, batch_size, True)
    for _ in range(epochs):
        autoencoder.train()
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = autoencoder(x)
            loss = F.mse_loss(reconstruction, x)
            loss.backward()
            optimizer.step()
    return autoencoder


@torch.no_grad()
def _accuracy(model, x, y, batch_size=512):
    model.eval()
    correct = 0
    for start in range(0, len(x), batch_size):
        bx = x[start : start + batch_size]
        by = y[start : start + batch_size]
        correct += int(model(bx).argmax(1).eq(by).sum().item())
    return correct / max(len(x), 1)


def _train_quantized_image(
    q_model,
    fp_model,
    autoencoder,
    bundle,
    device,
    method,
    target_class,
    epochs=4,
    batch_size=128,
    eta=1.0,
    target_avg_bits=4.0,
    teacher_steps=2,
    preserve_zeros=False,
):
    q_model.to(device)
    pruning_masks = [(layer.weight, layer.weight.detach().ne(0)) for layer in iter_quant_layers(q_model)] if preserve_zeros else []
    fp_model.to(device).eval()
    autoencoder.to(device).eval()
    for parameter in fp_model.parameters():
        parameter.requires_grad_(False)
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    fixed_bit = 4 if method in {"lsq", "pact", "cfq_uniform"} else None
    use_cf = method in {"cfq", "cfq_uniform"}
    optimizer = torch.optim.AdamW(q_model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = _loader(bundle.x_train, bundle.y_train, batch_size, True)
    solver = LatentRecourseSolver(steps=teacher_steps, step_size=0.12, restarts=1)
    for epoch in range(epochs):
        q_model.train()
        q_model.temperature = max(0.25, 5.0 * (0.25 / 5.0) ** (epoch / max(epochs - 1, 1)))
        q_model.hard = True
        q_model.stochastic = fixed_bit is None
        q_model.fixed_bit = fixed_bit
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mask = y.ne(target_class)
            cf = x
            success = torch.zeros_like(mask)
            if use_cf and mask.any():
                result = solver.solve(fp_model, autoencoder, x[mask], target_class)
                cf = x.clone()
                cf[mask] = result.counterfactual
                success[mask] = result.success
            optimizer.zero_grad(set_to_none=True)
            task_loss = F.cross_entropy(q_model(x), y)
            loss = task_loss
            if use_cf and success.any():
                targets = torch.full((success.sum(),), target_class, device=device, dtype=torch.long)
                loss = loss + eta * F.cross_entropy(q_model(cf[success]), targets)
            budget_penalty = F.relu(bit_cost(q_model) - target_avg_bits).square()
            loss = loss + 0.01 * budget_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_model.parameters(), 10.0)
            optimizer.step()
            with torch.no_grad():
                for weight, mask in pruning_masks:
                    weight.mul_(mask)
    q_model.temperature = 0.25
    q_model.hard = True
    q_model.stochastic = False
    q_model.fixed_bit = fixed_bit


def _calibrate_image_ptq(q_model, fp_model, calibration_x, cf_x, device, bit, epochs=3):
    q_model.to(device)
    fp_model.to(device).eval()
    q_model.fixed_bit = bit
    q_model.stochastic = False
    for name, parameter in q_model.named_parameters():
        parameter.requires_grad_(any(token in name for token in ("raw_steps", "raw_clips")))
    trainable = [parameter for parameter in q_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=5e-3)
    inputs = torch.cat([calibration_x, cf_x], dim=0) if cf_x is not None else calibration_x
    loader = DataLoader(TensorDataset(inputs.cpu()), batch_size=128, shuffle=True)
    for _ in range(epochs):
        q_model.train()
        for (x,) in loader:
            x = x.to(device)
            with torch.no_grad():
                teacher_logits = fp_model(x)
            loss = F.mse_loss(q_model(x), teacher_logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    for parameter in q_model.parameters():
        parameter.requires_grad_(True)
    q_model.eval()


def _latent_metrics(fp_model, q_model, autoencoder, x, y, target_class, eval_examples=128):
    full_accuracy = _accuracy(q_model, x, y)
    x = x[:eval_examples]
    y = y[:eval_examples]
    target_mask = y.ne(target_class)
    x = x[target_mask]
    y = y[target_mask]
    solver = LatentRecourseSolver(steps=40, step_size=0.08, restarts=2)
    with torch.enable_grad():
        fp_result = solver.solve(fp_model, autoencoder, x, target_class)
        q_result = fp_result if q_model is fp_model else solver.solve(q_model, autoencoder, x, target_class)
    target = torch.full((x.shape[0],), target_class, device=x.device, dtype=torch.long)
    with torch.no_grad():
        transferred = q_model(fp_result.counterfactual).argmax(1).eq(target)
        valid = fp_result.success
        vd = (~transferred[valid]).float().mean()
        both = fp_result.success & q_result.success & (fp_result.cost > 1e-8)
        crg_values = (q_result.cost - fp_result.cost) / fp_result.cost.clamp_min(1e-8)
        dirs = F.cosine_similarity(fp_result.delta, q_result.delta, dim=1)
        return {
            "accuracy": full_accuracy,
            "validity_drop": float(vd.item()),
            "crg": float(crg_values[both].mean().item()) if both.any() else float("nan"),
            "direction_similarity": float(dirs[both].mean().item()) if both.any() else float("nan"),
            "fp_feasible_rate": float(fp_result.success.float().mean().item()),
            "q_feasible_rate": float(q_result.success.float().mean().item()),
        }


def run_image_experiment(
    dataset: str = "mnist",
    method: str = "cfq",
    output_dir: str | Path = "results/image",
    data_root: str | Path = "data",
    target_class: int = 0,
    max_train: int | None = 5000,
    max_test: int | None = 1000,
    classifier_epochs: int = 5,
    autoencoder_epochs: int = 5,
    qat_epochs: int = 4,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, Any]:
    method = method.lower()
    supported = {"fp32", "lsq", "pact", "mixedprec", "cfq", "cfq_uniform", "prune_quant", "ptq4", "ptq8", "cfptq"}
    if method not in supported:
        raise ValueError(f"Unknown image method {method!r}; choose from {sorted(supported)}")
    if not 0 <= target_class < 10 or min(classifier_epochs, autoencoder_epochs, qat_epochs) < 1:
        raise ValueError("target_class must be in [0, 9] and training epochs must be positive")
    seed_everything(seed)
    target_device = resolve_device(device)
    bundle = load_image_dataset(dataset, data_root, max_train=max_train, max_test=max_test, seed=seed)
    fp_model = _train_classifier(SmallCNN(bundle.num_classes), bundle, target_device, epochs=classifier_epochs)
    autoencoder = _train_autoencoder(ConvAutoencoder(32), bundle, target_device, epochs=autoencoder_epochs)
    autoencoder.eval()
    q_model = QuantSmallCNN(bundle.num_classes, quantize_activations=method != "lsq")
    copy_fp_to_quantized(fp_model, q_model)
    method = method.lower()

    if method == "fp32":
        q_model = fp_model
    elif method == "prune_quant":
        magnitude_prune(fp_model, 0.30)
        copy_fp_to_quantized(fp_model, q_model)
        method_for_training = "mixedprec"
        _train_quantized_image(q_model, fp_model, autoencoder, bundle, target_device, method_for_training, target_class, epochs=qat_epochs, preserve_zeros=True)
    elif method in {"ptq4", "ptq8", "cfptq"}:
        bit = 4 if method != "ptq8" else 8
        cf_x = None
        if method == "cfptq":
            calibration = bundle.x_val.to(target_device)
            calibration_y = bundle.y_val.to(target_device)
            mask = calibration_y.ne(target_class)
            solver = LatentRecourseSolver(steps=2, step_size=0.12, restarts=1)
            with torch.enable_grad():
                result = solver.solve(fp_model, autoencoder, calibration[mask], target_class)
            cf_x = result.counterfactual[result.success]
        else:
            calibration = bundle.x_val.to(target_device)
        _calibrate_image_ptq(q_model, fp_model, calibration, cf_x, target_device, bit)
    else:
        _train_quantized_image(q_model, fp_model, autoencoder, bundle, target_device, method, target_class, epochs=qat_epochs)

    fp_model.eval()
    q_model.eval()
    metrics = _latent_metrics(
        fp_model,
        q_model,
        autoencoder,
        bundle.x_test.to(target_device),
        bundle.y_test.to(target_device),
        target_class,
    )
    result = {
        "dataset": bundle.name,
        "method": method,
        "target_class": target_class,
        "fp_accuracy": _accuracy(fp_model, bundle.x_test.to(target_device), bundle.y_test.to(target_device)),
        "metrics": metrics,
        "bit_allocation": hard_bit_allocation(q_model),
        "average_bits": float(bit_cost(q_model).detach().cpu().item()),
        "note": "CelebA requires a supplied SemanticEditor; MNIST/Fashion use the executable latent-autoencoder protocol.",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_json(result, output / "metrics.json")
    save_quantization_settings(q_model, output)
    save_json({"dataset": dataset, "method": method, "target_class": target_class,
               "seed": seed, "max_train": max_train, "max_test": max_test,
               "classifier_epochs": classifier_epochs, "autoencoder_epochs": autoencoder_epochs,
               "qat_epochs": qat_epochs}, output / "config.json")
    torch.save(fp_model.state_dict(), output / "fp_model.pt")
    torch.save(q_model.state_dict(), output / "quantized_model.pt")
    torch.save(autoencoder.state_dict(), output / "autoencoder.pt")
    return result
