import numpy as np
import pandas as pd
import pytest
import torch

from cfq.cli import config_from_mapping
from cfq.config import ExperimentConfig, prepare_config
from cfq.data.tabular import RawDatasetSpec, _binary_target, load_tabular_dataset
from cfq.models import make_tabular_model, make_quant_tabular_model


def test_numeric_float_labels_and_unknown_polarity():
    assert _binary_target(pd.Series([0., 1., 0.]), (1,)).tolist() == [0, 1, 0]
    with pytest.raises(ValueError, match='favorable'):
        _binary_target(pd.Series(['bad', 'good']), ('yes',))


def test_preprocessing_imputes_from_training_only(monkeypatch):
    frame = pd.DataFrame({'value': np.arange(100, dtype=float), 'empty': np.nan,
                          'category': ['a', None, 'b', 'a'] * 25, 'target': [0, 1] * 50})
    frame.loc[frame.index % 3 == 0, 'value'] = np.nan
    spec = RawDatasetSpec('test', None, ('target',), (1,), (), None, 3, .2)
    monkeypatch.setattr('cfq.data.tabular._load_raw', lambda *args: (frame.copy(), spec))
    bundle = load_tabular_dataset('test')
    imputer = bundle.metadata['preprocessor'].named_transformers_['numerical'].named_steps['imputer']
    from sklearn.model_selection import train_test_split
    train, _ = train_test_split(np.arange(100), test_size=.2, random_state=42, stratify=frame.target)
    train, _ = train_test_split(train, test_size=.15, random_state=42, stratify=frame.target.iloc[train])
    assert imputer.statistics_[0] == frame.value.iloc[train].median()
    assert torch.isfinite(bundle.x_train).all()
    assert torch.isfinite(bundle.x_test).all()
    assert 'category_missing' in bundle.feature_names


def test_subsample_keeps_original_test_fraction(monkeypatch):
    frame = pd.DataFrame({'value': np.arange(1000), 'target': [0, 1] * 500})
    spec = RawDatasetSpec('test', None, ('target',), (1,), (), None, 1, 200)
    monkeypatch.setattr('cfq.data.tabular._load_raw', lambda *args: (frame.copy(), spec))
    bundle = load_tabular_dataset('test', max_samples=300)
    assert len(bundle.x_test) == 60


def test_unknown_settings_are_not_silently_ignored():
    with pytest.raises(ValueError, match='Unknown'):
        config_from_mapping({'train': {'epohs_fp': 3}})
    with pytest.raises(ValueError, match='Unknown method'):
        prepare_config(ExperimentConfig(method='cfqq'))


def test_dropout_is_used_by_both_model_factories():
    for factory in (make_tabular_model, make_quant_tabular_model):
        model = factory('mlp', 3, hidden_dims=(4,), dropout=.3)
        assert any(isinstance(layer, torch.nn.Dropout) and layer.p == .3 for layer in model.modules())


@pytest.mark.parametrize('dataset,target_name,columns,labels', [
    ('bank', 'Class', ['V1', 'V2', 'V3'], ['1', '2']),
    ('default', 'y', ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'], ['0', '1']),
])
def test_openml_target_is_never_duplicated_as_feature(dataset, target_name, columns, labels, monkeypatch):
    from types import SimpleNamespace
    from cfq.data.tabular import _load_raw
    from pathlib import Path
    features = pd.DataFrame({column: [1, 2] * 30 for column in columns})
    target = pd.Series(labels * 30, name=target_name)
    raw = features.copy()
    raw[target_name] = target
    bunch = SimpleNamespace(data=features, target=target, frame=raw)
    monkeypatch.setattr('cfq.data.tabular.fetch_openml', lambda *args, **kwargs: bunch)
    frame, spec = _load_raw(dataset, Path('unused'))
    assert frame.shape[1] == features.shape[1] + 1
    assert target_name not in frame.columns
    if dataset == 'bank':
        assert _binary_target(frame.y, spec.positive_values)[:2].tolist() == [0, 1]
        assert {'age', 'marital'} <= set(frame.columns)
    else:
        assert {'SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0'} <= set(frame.columns)
