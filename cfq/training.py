from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import RecourseConfig, TrainConfig, QuantConfig
from .costs import recourse_cost
from .metrics import accuracy
from .quantization import bit_cost, configure_quantization, iter_quant_layers
from .recourse import PGDRecourseSolver, target_margin


@dataclass
class TrainHistory:
    losses: list[float]
    validation_accuracy: list[float]
    elapsed_seconds: float
    best_epoch: int


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


def _temperature(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    ratio = epoch / (epochs - 1)
    return float(start * (end / start) ** ratio)


def train_supervised(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
) -> TrainHistory:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = make_loader(x_train, y_train, config.batch_size, True, config.seed)
    losses: list[float] = []
    val_scores: list[float] = []
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    best_epoch = 0
    patience = 0
    start_time = time.perf_counter()

    for epoch in range(config.epochs_fp):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_x.shape[0]
            total_examples += batch_x.shape[0]
        score = accuracy(model, x_val.to(device), y_val.to(device))
        losses.append(total_loss / max(total_examples, 1))
        val_scores.append(score)
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= config.early_stopping_patience:
            break
    model.load_state_dict(best_state)
    return TrainHistory(losses, val_scores, time.perf_counter() - start_time, best_epoch)


def train_recourse_margin_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    action_set,
    feature_weights: torch.Tensor,
    target_label: int,
    train_config: TrainConfig,
    recourse_config: RecourseConfig,
    device: torch.device,
    margin_weight: float = 0.5,
) -> TrainHistory:
    """R-Margin baseline: train FP model with a margin at its recourse points."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    loader = make_loader(x_train, y_train, train_config.batch_size, True, train_config.seed)
    solver = PGDRecourseSolver(
        steps=recourse_config.train_steps,
        step_size=recourse_config.train_step_size,
        restarts=1,
        cost_kind=recourse_config.cost_kind,
        cost_weight=recourse_config.cost_weight,
        mixed_l1=recourse_config.mixed_l1,
        mixed_l2=recourse_config.mixed_l2,
        margin=recourse_config.margin,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_score = -1.0
    best_epoch = 0
    patience = 0
    losses: list[float] = []
    scores: list[float] = []
    start = time.perf_counter()
    local_action = action_set.to(device)
    local_weights = feature_weights.to(device)

    for epoch in range(train_config.epochs_fp):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            target = torch.full_like(batch_y, target_label)
            with torch.enable_grad():
                result = solver.solve(model, batch_x, target, local_action, local_weights, detach_result=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            factual_loss = F.cross_entropy(logits, batch_y)
            cf_logits = model(batch_x + result.delta)
            margin_loss = F.relu(train_config.hinge_gamma - target_margin(cf_logits, target))
            mask = result.success.float()
            margin_loss = (margin_loss * mask).sum() / mask.sum().clamp_min(1.0)
            loss = factual_loss + margin_weight * margin_loss
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch_x.shape[0]
            count += batch_x.shape[0]
        score = accuracy(model, x_val.to(device), y_val.to(device))
        losses.append(total / max(count, 1))
        scores.append(score)
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= train_config.early_stopping_patience:
            break
    model.load_state_dict(best_state)
    return TrainHistory(losses, scores, time.perf_counter() - start, best_epoch)


def train_quantized(
    q_model: nn.Module,
    teacher: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    action_set,
    feature_weights: torch.Tensor,
    target_label: int,
    method: str,
    train_config: TrainConfig,
    recourse_config: RecourseConfig,
    quant_config: QuantConfig,
    device: torch.device,
    teacher_noise: float = 0.0,
    recourse_batch_fraction: float = 1.0,
    preserve_zeros: bool = False,
) -> TrainHistory:
    """Train LSQ/PACT/mixed-precision/CFQ variants with one implementation.

    Methods:
      - ``lsq``: uniform-bit weight QAT, no activation quantization expected.
      - ``pact``: uniform-bit weight+activation QAT.
      - ``mixedprec``: learned bit allocation with task and budget losses.
      - ``cfq``: mixed precision plus teacher-counterfactual validity loss.
      - ``cfq_uniform``: CFQ objective at a fixed uniform bitwidth.
      - ``cfq_match``: CFQ with differentiable student-action matching.
    """
    method = method.lower()
    q_model.to(device)
    pruning_masks = [(layer.weight, layer.weight.detach().ne(0)) for layer in iter_quant_layers(q_model)] if preserve_zeros else []
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    fixed_bit = quant_config.uniform_bit if method in {"lsq", "pact", "cfq_uniform"} or not quant_config.mixed_precision else None
    if method == "lsq" and hasattr(q_model, "quantize_activations"):
        q_model.quantize_activations = False
    use_cf = method.startswith("cfq")
    use_match = method == "cfq_match" or train_config.match_alpha1 > 0 or train_config.match_alpha2 > 0
    optimizer = torch.optim.AdamW(q_model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    loader = make_loader(x_train, y_train, train_config.batch_size, True, train_config.seed)
    teacher_solver = PGDRecourseSolver(
        steps=recourse_config.train_steps,
        step_size=recourse_config.train_step_size,
        restarts=recourse_config.train_restarts,
        cost_kind=recourse_config.cost_kind,
        cost_weight=recourse_config.cost_weight,
        mixed_l1=recourse_config.mixed_l1,
        mixed_l2=recourse_config.mixed_l2,
        margin=recourse_config.margin,
    )
    student_solver = PGDRecourseSolver(
        steps=min(recourse_config.train_steps, 3),
        step_size=recourse_config.train_step_size,
        restarts=1,
        cost_kind=recourse_config.cost_kind,
        cost_weight=recourse_config.cost_weight,
        mixed_l1=recourse_config.mixed_l1,
        mixed_l2=recourse_config.mixed_l2,
        margin=recourse_config.margin,
    )
    local_action = action_set.to(device)
    local_weights = feature_weights.to(device)
    best_state = copy.deepcopy(q_model.state_dict())
    best_score = -float("inf")
    best_epoch = 0
    patience = 0
    losses: list[float] = []
    val_scores: list[float] = []
    start_time = time.perf_counter()

    for epoch in range(train_config.epochs_qat):
        q_model.train()
        temperature = _temperature(
            epoch,
            train_config.epochs_qat,
            quant_config.temperature_start,
            quant_config.temperature_end,
        )
        setattr(q_model, "temperature", temperature)
        setattr(q_model, "hard", True)
        setattr(q_model, "stochastic", fixed_bit is None)
        setattr(q_model, "fixed_bit", fixed_bit)
        total_loss = 0.0
        total_examples = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            target = torch.full_like(batch_y, target_label)
            selected_mask = torch.rand(batch_x.shape[0], device=device) < recourse_batch_fraction
            if recourse_batch_fraction > 0 and not selected_mask.any():
                selected_mask[0] = True

            teacher_delta = torch.zeros_like(batch_x)
            teacher_success = torch.zeros(batch_x.shape[0], dtype=torch.bool, device=device)
            if use_cf and selected_mask.any():
                selected_x = batch_x[selected_mask]
                selected_target = target[selected_mask]
                with torch.enable_grad():
                    result = teacher_solver.solve(
                        teacher,
                        selected_x,
                        selected_target,
                        local_action,
                        local_weights,
                        detach_result=True,
                    )
                delta = result.delta
                if teacher_noise > 0:
                    delta = delta + teacher_noise * torch.randn_like(delta)
                    delta = local_action.project(selected_x, delta)
                    with torch.no_grad():
                        result.success = teacher(selected_x + delta).argmax(1).eq(selected_target)
                        result.success &= local_action.is_feasible(selected_x, delta)
                teacher_delta[selected_mask] = delta
                teacher_success[selected_mask] = result.success

            optimizer.zero_grad(set_to_none=True)
            factual_logits = q_model(batch_x)
            task_loss = F.cross_entropy(factual_logits, batch_y)
            total = task_loss

            if use_cf:
                cf_logits = q_model(batch_x + teacher_delta)
                validity_per_example = F.cross_entropy(cf_logits, target, reduction="none")
                valid_mask = selected_mask & teacher_success
                validity_loss = (validity_per_example * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)
                total = total + train_config.eta * validity_loss
                if train_config.hinge_beta > 0:
                    hinge = F.relu(train_config.hinge_gamma - target_margin(cf_logits, target))
                    hinge_loss = (hinge * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)
                    total = total + train_config.hinge_beta * hinge_loss

                if use_match and valid_mask.any():
                    # Action matching differentiates through one deployment policy;
                    # resampling Gumbel noise at every inner step changes the model.
                    q_model.stochastic = False
                    student_result = student_solver.solve(
                        q_model,
                        batch_x[valid_mask],
                        target[valid_mask],
                        local_action,
                        local_weights,
                        create_graph=True,
                        detach_result=False,
                    )
                    q_model.stochastic = fixed_bit is None
                    teacher_selected = teacher_delta[valid_mask]
                    match_direction = (student_result.delta - teacher_selected).abs().sum(dim=-1).mean()
                    student_cost = recourse_cost(student_result.delta, local_weights, recourse_config.cost_kind,
                                                recourse_config.mixed_l1, recourse_config.mixed_l2, smooth_eps=1e-6)
                    teacher_cost = recourse_cost(teacher_selected, local_weights, recourse_config.cost_kind,
                                                recourse_config.mixed_l1, recourse_config.mixed_l2, smooth_eps=1e-6)
                    match_cost = (student_cost - teacher_cost).abs().mean()
                    total = total + train_config.match_alpha1 * match_direction + train_config.match_alpha2 * match_cost

            average_bits = bit_cost(q_model)
            budget_violation = F.relu(average_bits - quant_config.target_avg_bits)
            total = total + train_config.bit_lambda * budget_violation.square()
            total.backward()
            torch.nn.utils.clip_grad_norm_(q_model.parameters(), max_norm=10.0)
            optimizer.step()
            with torch.no_grad():
                for weight, mask in pruning_masks:
                    weight.mul_(mask)
            total_loss += float(total.item()) * batch_x.shape[0]
            total_examples += batch_x.shape[0]

        setattr(q_model, "temperature", quant_config.temperature_end)
        setattr(q_model, "hard", True)
        setattr(q_model, "stochastic", False)
        setattr(q_model, "fixed_bit", fixed_bit)
        score = accuracy(q_model, x_val.to(device), y_val.to(device))
        losses.append(total_loss / max(total_examples, 1))
        val_scores.append(score)
        current_bits = float(bit_cost(q_model).detach().cpu().item())
        budget_ok = current_bits <= quant_config.target_avg_bits + quant_config.budget_tolerance
        selection_score = score - (0.0 if budget_ok else 0.1 * (current_bits - quant_config.target_avg_bits))
        if selection_score > best_score + 1e-6:
            best_score = selection_score
            best_state = copy.deepcopy(q_model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
        if patience >= train_config.early_stopping_patience:
            break

    q_model.load_state_dict(best_state)
    setattr(q_model, "temperature", quant_config.temperature_end)
    setattr(q_model, "hard", True)
    setattr(q_model, "stochastic", False)
    setattr(q_model, "fixed_bit", fixed_bit)
    configure_quantization(q_model, quant_config.temperature_end, hard=True, stochastic=False, fixed_bit=fixed_bit)
    return TrainHistory(losses, val_scores, time.perf_counter() - start_time, best_epoch)


def distill_model(
    student: nn.Module,
    teacher: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> TrainHistory:
    student.to(device)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = make_loader(x_train, y_train, config.batch_size, True, config.seed)
    losses, scores = [], []
    best_state = copy.deepcopy(student.state_dict())
    best_score, best_epoch, patience = -1.0, 0, 0
    start = time.perf_counter()
    for epoch in range(config.epochs_fp):
        student.train()
        total, count = 0.0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            with torch.no_grad():
                teacher_logits = teacher(bx)
            student_logits = student(bx)
            hard_loss = F.cross_entropy(student_logits, by)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * temperature**2
            loss = alpha * hard_loss + (1 - alpha) * soft_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * bx.shape[0]
            count += bx.shape[0]
        score = accuracy(student, x_val.to(device), y_val.to(device))
        losses.append(total / max(count, 1))
        scores.append(score)
        if score > best_score + 1e-6:
            best_score, best_epoch, patience = score, epoch, 0
            best_state = copy.deepcopy(student.state_dict())
        else:
            patience += 1
        if patience >= config.early_stopping_patience:
            break
    student.load_state_dict(best_state)
    return TrainHistory(losses, scores, time.perf_counter() - start, best_epoch)


@torch.no_grad()
def magnitude_prune(model: nn.Module, fraction: float = 0.3) -> None:
    if not 0 <= fraction <= 1:
        raise ValueError("Pruning fraction must be in [0, 1]")
    if fraction == 0:
        return
    weights = [parameter.abs().flatten() for name, parameter in model.named_parameters() if "weight" in name and parameter.ndim > 1]
    if not weights:
        return
    threshold = torch.quantile(torch.cat(weights), fraction)
    for name, parameter in model.named_parameters():
        if "weight" in name and parameter.ndim > 1:
            parameter.mul_(parameter.abs() > threshold)


def train_recourse_consistency_model(
    model: nn.Module,
    peer: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    action_set,
    feature_weights: torch.Tensor,
    target_label: int,
    train_config: TrainConfig,
    recourse_config: RecourseConfig,
    device: torch.device,
    consistency_weight: float = 0.5,
) -> TrainHistory:
    """R-Consistency baseline using two jointly trained plausible predictors."""
    model.to(device)
    peer.to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(peer.parameters()),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    loader = make_loader(x_train, y_train, train_config.batch_size, True, train_config.seed)
    solver = PGDRecourseSolver(
        steps=recourse_config.train_steps,
        step_size=recourse_config.train_step_size,
        restarts=1,
        cost_kind=recourse_config.cost_kind,
        cost_weight=recourse_config.cost_weight,
        mixed_l1=recourse_config.mixed_l1,
        mixed_l2=recourse_config.mixed_l2,
        margin=recourse_config.margin,
    )
    local_action = action_set.to(device)
    local_weights = feature_weights.to(device)
    losses, scores = [], []
    best_state = copy.deepcopy(model.state_dict())
    best_score, best_epoch, patience = -1.0, 0, 0
    start = time.perf_counter()
    for epoch in range(train_config.epochs_fp):
        model.train()
        peer.train()
        total, count = 0.0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            target = torch.full_like(by, target_label)
            with torch.enable_grad():
                teacher_result = solver.solve(model, bx, target, local_action, local_weights, detach_result=True)
            optimizer.zero_grad(set_to_none=True)
            logits_a = model(bx)
            logits_b = peer(bx)
            task = 0.5 * (F.cross_entropy(logits_a, by) + F.cross_entropy(logits_b, by))
            cf_x = bx + teacher_result.delta
            cf_a = model(cf_x)
            cf_b = peer(cf_x)
            probs_a = F.log_softmax(cf_a, dim=1)
            probs_b = F.log_softmax(cf_b, dim=1)
            target_a = F.softmax(cf_a.detach(), dim=1)
            target_b = F.softmax(cf_b.detach(), dim=1)
            consistency = 0.5 * (
                F.kl_div(probs_a, target_b, reduction="batchmean")
                + F.kl_div(probs_b, target_a, reduction="batchmean")
            )
            loss = task + consistency_weight * consistency
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * bx.shape[0]
            count += bx.shape[0]
        score = accuracy(model, x_val.to(device), y_val.to(device))
        losses.append(total / max(count, 1))
        scores.append(score)
        if score > best_score + 1e-6:
            best_score, best_epoch, patience = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
        if patience >= train_config.early_stopping_patience:
            break
    model.load_state_dict(best_state)
    return TrainHistory(losses, scores, time.perf_counter() - start, best_epoch)
