from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..constraints import ActionSet
from ..costs import feature_weights_from_std


@dataclass(frozen=True)
class RawDatasetSpec:
    name: str
    openml_name: str | None
    target_candidates: tuple[str, ...]
    positive_values: tuple[Any, ...]
    immutable_columns: tuple[str, ...]
    subgroup_column: str | None
    sparsity: int
    test_size: int | float
    primary_cost: str = "l1"
    categorical_columns: tuple[str, ...] = ()
    ordinal_columns: tuple[str, ...] = ()


DATASET_SPECS: dict[str, RawDatasetSpec] = {
    "adult": RawDatasetSpec(
        name="adult",
        openml_name="adult",
        target_candidates=("class", "income"),
        positive_values=(">50K", ">50K."),
        immutable_columns=("age", "sex", "race", "native-country"),
        subgroup_column="sex",
        sparsity=5,
        test_size=16281,
    ),
    "german": RawDatasetSpec(
        name="german",
        openml_name="credit-g",
        target_candidates=("class",),
        positive_values=("good", 1, "1"),
        immutable_columns=("age", "personal_status", "foreign_worker"),
        subgroup_column="personal_status",
        sparsity=4,
        test_size=300,
    ),
    "bank": RawDatasetSpec(
        name="bank",
        openml_name="bank-marketing",
        target_candidates=("y", "class"),
        positive_values=("yes", 2, "2"),
        immutable_columns=("age",),
        subgroup_column="marital",
        sparsity=5,
        test_size=9043,
    ),
    "default": RawDatasetSpec(
        name="default",
        openml_name="default-of-credit-card-clients",
        target_candidates=("default payment next month", "default_payment_next_month", "class"),
        positive_values=(0, "0", "no"),
        immutable_columns=("SEX", "sex"),
        subgroup_column="SEX",
        sparsity=4,
        test_size=6000,
        primary_cost="l2",
        categorical_columns=("SEX", "EDUCATION", "MARRIAGE"),
        ordinal_columns=("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
    ),
    "compas": RawDatasetSpec(
        name="compas",
        openml_name=None,
        target_candidates=("favorable",),
        positive_values=(1,),
        immutable_columns=("age", "sex", "race"),
        subgroup_column="race",
        sparsity=3,
        test_size=0.30,
    ),
}


@dataclass
class DatasetBundle:
    name: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    action_set: ActionSet
    feature_weights: torch.Tensor
    feature_names: tuple[str, ...]
    target_label: int = 1
    group_train: torch.Tensor | None = None
    group_val: torch.Tensor | None = None
    group_test: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    @property
    def input_dim(self) -> int:
        return int(self.x_train.shape[1])


def _detect_target(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lower_map = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise KeyError(f"Could not find target column among {candidates}; columns={list(frame.columns)}")


def _load_compas(cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "compas-scores-two-years.csv"
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    if destination.exists():
        frame = pd.read_csv(destination)
    else:
        frame = pd.read_csv(url)
        frame.to_csv(destination, index=False)
    frame = frame.loc[
        (frame["days_b_screening_arrest"].between(-30, 30))
        & (frame["is_recid"] != -1)
        & (frame["c_charge_degree"] != "O")
        & frame["score_text"].notna()
        & (frame["score_text"] != "N/A")
    ].copy()
    columns = [
        "age",
        "sex",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_charge_degree",
        "two_year_recid",
    ]
    frame = frame[columns].dropna()
    frame["favorable"] = (frame.pop("two_year_recid") == 0).astype(int)
    return frame


def _load_raw(name: str, cache_dir: Path) -> tuple[pd.DataFrame, RawDatasetSpec]:
    key = name.lower()
    if key == "synthetic":
        x, y = make_classification(
            n_samples=2400,
            n_features=12,
            n_informative=8,
            n_redundant=2,
            class_sep=1.2,
            random_state=42,
        )
        frame = pd.DataFrame(x, columns=[f"x{index}" for index in range(x.shape[1])])
        frame["target"] = y
        spec = RawDatasetSpec(
            name="synthetic",
            openml_name=None,
            target_candidates=("target",),
            positive_values=(1,),
            immutable_columns=("x0", "x1"),
            subgroup_column="x0",
            sparsity=4,
            test_size=0.2,
        )
        return frame, spec
    if key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {name!r}; choose from synthetic, {', '.join(DATASET_SPECS)}")
    spec = DATASET_SPECS[key]
    if key == "compas":
        return _load_compas(cache_dir), spec
    assert spec.openml_name is not None
    bunch = fetch_openml(spec.openml_name, version=1, as_frame=True, parser="auto", data_home=str(cache_dir))
    # .frame includes the target under its source name (e.g. Class or y).
    # Adding a second alias to it leaks labels into the feature matrix.
    frame = bunch.data.copy()
    if bunch.target is None or not isinstance(bunch.target, pd.Series):
        raise ValueError(f"Expected one target column for {name}")
    if key == "bank":
        bank_names = ("age", "job", "marital", "education", "default", "balance",
                      "housing", "loan", "contact", "day", "month", "duration",
                      "campaign", "pdays", "previous", "poutcome")
        frame = frame.rename(columns={f"V{i}": column for i, column in enumerate(bank_names, 1)})
    elif key == "default":
        default_names = ("LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
                         "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                         *(f"BILL_AMT{i}" for i in range(1, 7)),
                         *(f"PAY_AMT{i}" for i in range(1, 7)))
        frame = frame.rename(columns={f"x{i}": column for i, column in enumerate(default_names, 1)})
    frame[spec.target_candidates[0]] = bunch.target
    return frame, spec


def _binary_target(series: pd.Series, positive_values: tuple[Any, ...]) -> np.ndarray:
    if series.empty or series.isna().any():
        raise ValueError("Binary targets must be nonempty and cannot contain missing values")
    def normalize(value):
        text = str(value).strip()
        try:
            number = float(text)
            return str(int(number)) if np.isfinite(number) and number.is_integer() else text
        except ValueError:
            return text
    normalized = series.map(normalize)
    positive_strings = {normalize(value) for value in positive_values}
    values = normalized.isin(positive_strings).astype(np.int64).to_numpy()
    unique = list(pd.unique(normalized))
    if len(unique) != 2 or values.min() == values.max():
        raise ValueError(f"Expected two target classes with a known favorable class {positive_values}; found {unique[:10]}")
    return values


def _make_subgroup(series: pd.Series | None) -> np.ndarray | None:
    if series is None:
        return None
    if pd.api.types.is_numeric_dtype(series):
        threshold = float(series.median())
        return (pd.to_numeric(series, errors="coerce").fillna(threshold) > threshold).astype(np.int64).to_numpy()
    codes, _ = pd.factorize(series.astype(str).fillna("missing"), sort=True)
    return codes.astype(np.int64)


def _resolve_columns(frame: pd.DataFrame, requested: tuple[str, ...]) -> tuple[str, ...]:
    lower = {str(column).lower(): str(column) for column in frame.columns}
    resolved = []
    for column in requested:
        if column in frame.columns:
            resolved.append(column)
        elif column.lower() in lower:
            resolved.append(lower[column.lower()])
    return tuple(resolved)


def load_tabular_dataset(
    name: str,
    cache_dir: str | Path = "data",
    seed: int = 42,
    validation_fraction: float = 0.15,
    max_samples: int | None = None,
) -> DatasetBundle:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if max_samples is not None and max_samples < 10:
        raise ValueError("max_samples must be at least 10 for stratified train/validation/test splits")
    frame, spec = _load_raw(name, Path(cache_dir))
    source_rows = len(frame)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    target_column = _detect_target(frame, spec.target_candidates)
    target = _binary_target(frame.pop(target_column), spec.positive_values)
    if max_samples is not None and len(frame) > max_samples:
        selected, _ = train_test_split(np.arange(len(frame)), train_size=max_samples,
                                       random_state=seed, stratify=target)
        frame = frame.iloc[selected].reset_index(drop=True)
        target = target[selected]

    subgroup_column = None
    if spec.subgroup_column is not None:
        resolved_group = _resolve_columns(frame, (spec.subgroup_column,))
        subgroup_column = resolved_group[0] if resolved_group else None
    group = _make_subgroup(frame[subgroup_column] if subgroup_column else None)

    declared_categorical = _resolve_columns(frame, spec.categorical_columns)
    categorical = [column for column in frame.columns if column in declared_categorical
                   or not pd.api.types.is_numeric_dtype(frame[column])]
    numerical = [column for column in frame.columns if column not in categorical]
    for column in categorical:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), "missing").astype(str)
    for column in numerical:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    indices = np.arange(len(frame))
    requested_test_size = spec.test_size
    if isinstance(requested_test_size, int) and len(frame) < source_rows:
        requested_test_size = requested_test_size / source_rows
    train_indices, test_indices = train_test_split(
        indices,
        test_size=requested_test_size,
        random_state=seed,
        stratify=target,
    )
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=target[train_indices],
    )

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    transformer = ColumnTransformer(
        [("categorical", encoder, categorical), ("numerical", numeric_pipeline, numerical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    x_train_np = transformer.fit_transform(frame.iloc[train_indices]).astype(np.float32)
    x_val_np = transformer.transform(frame.iloc[val_indices]).astype(np.float32)
    x_test_np = transformer.transform(frame.iloc[test_indices]).astype(np.float32)
    feature_names = tuple(str(name) for name in transformer.get_feature_names_out())

    categorical_groups: list[tuple[int, ...]] = []
    offset = 0
    categorical_index_by_column: dict[str, tuple[int, ...]] = {}
    if categorical:
        fitted_encoder: OneHotEncoder = transformer.named_transformers_["categorical"]
        for column, categories in zip(categorical, fitted_encoder.categories_):
            indices_for_group = tuple(range(offset, offset + len(categories)))
            categorical_groups.append(indices_for_group)
            categorical_index_by_column[column] = indices_for_group
            offset += len(categories)
    numerical_index_by_column = {column: offset + index for index, column in enumerate(numerical)}
    ordinal_domains = {}
    if numerical:
        scaler = transformer.named_transformers_["numerical"].named_steps["scaler"]
        for column in _resolve_columns(frame, spec.ordinal_columns):
            if column in numerical_index_by_column:
                position = numerical.index(column)
                domain = sorted(frame.iloc[train_indices][column].dropna().unique())
                if domain:
                    ordinal_domains[numerical_index_by_column[column]] = tuple(
                        float((value - scaler.mean_[position]) / scaler.scale_[position]) for value in domain
                    )

    immutable_columns = _resolve_columns(frame, spec.immutable_columns)
    immutable_indices: list[int] = []
    for column in immutable_columns:
        if column in categorical_index_by_column:
            immutable_indices.extend(categorical_index_by_column[column])
        elif column in numerical_index_by_column:
            immutable_indices.append(numerical_index_by_column[column])

    lower = np.nanmin(x_train_np, axis=0)
    upper = np.nanmax(x_train_np, axis=0)
    span = np.maximum(upper - lower, 1e-3)
    lower = lower - 0.05 * span
    upper = upper + 0.05 * span
    for group_indices in categorical_groups:
        lower[list(group_indices)] = 0.0
        upper[list(group_indices)] = 1.0

    x_train = torch.from_numpy(x_train_np)
    x_val = torch.from_numpy(x_val_np)
    x_test = torch.from_numpy(x_test_np)
    action_set = ActionSet(
        lower=torch.from_numpy(lower.astype(np.float32)),
        upper=torch.from_numpy(upper.astype(np.float32)),
        immutable=tuple(sorted(set(immutable_indices))),
        sparsity=min(spec.sparsity, x_train_np.shape[1]),
        categorical_groups=tuple(categorical_groups),
        ordinal_domains=ordinal_domains,
    )
    weights = feature_weights_from_std(x_train)

    def tensor_group(split_indices: np.ndarray) -> torch.Tensor | None:
        return None if group is None else torch.from_numpy(group[split_indices])

    return DatasetBundle(
        name=spec.name,
        x_train=x_train,
        y_train=torch.from_numpy(target[train_indices]),
        x_val=x_val,
        y_val=torch.from_numpy(target[val_indices]),
        x_test=x_test,
        y_test=torch.from_numpy(target[test_indices]),
        action_set=action_set,
        feature_weights=weights,
        feature_names=feature_names,
        target_label=1,
        group_train=tensor_group(train_indices),
        group_val=tensor_group(val_indices),
        group_test=tensor_group(test_indices),
        metadata={
            "source_rows": source_rows,
            "sampled_rows": len(frame),
            "categorical_columns": categorical,
            "numerical_columns": numerical,
            "immutable_columns_resolved": immutable_columns,
            "subgroup_column": subgroup_column,
            "primary_cost": spec.primary_cost,
            "preprocessor": transformer,
        },
    )
