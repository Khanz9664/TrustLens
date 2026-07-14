from types import SimpleNamespace

import numpy as np
import pytest

from trustlens import TrustReport, analyze
from trustlens.backends.registry import detect_framework, get_resolver
from trustlens.backends.types import UnsupportedModelError

# We only run these tests if transformers is available
transformers = pytest.importorskip("transformers")


def _make_text_classification_pipeline(id2label, outputs):
    """
    Build a lightweight double of a `TextClassificationPipeline` that subclasses
    the real class (so `type(model).__module__` starts with 'transformers' and
    `isinstance(model, TextClassificationPipeline)` is True) without invoking the
    real `__init__`, which requires a loaded model/tokenizer.
    """
    from transformers import TextClassificationPipeline

    class MockTextClassificationPipeline(TextClassificationPipeline):
        def __init__(self):
            self.task = "text-classification"
            self.model = SimpleNamespace(config=SimpleNamespace(id2label=id2label))

        def __call__(self, X, top_k=None):
            return outputs

    MockTextClassificationPipeline.__module__ = TextClassificationPipeline.__module__
    return MockTextClassificationPipeline()


def _make_non_classification_pipeline():
    """
    Build a double of a `TokenClassificationPipeline` (NER) — a real transformers
    pipeline class (so it's still detected as 'huggingface' via module prefix) but
    not a text-classification pipeline, so the resolver should reject it.
    """
    from transformers import TokenClassificationPipeline

    class MockTokenClassificationPipeline(TokenClassificationPipeline):
        def __init__(self):
            self.task = "ner"
            self.model = SimpleNamespace(config=SimpleNamespace())

        def __call__(self, X, **kwargs):
            return [[] for _ in X]

    MockTokenClassificationPipeline.__module__ = TokenClassificationPipeline.__module__
    return MockTokenClassificationPipeline()


def test_huggingface_detection():
    model = _make_text_classification_pipeline(
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        outputs=[[{"label": "NEGATIVE", "score": 0.5}, {"label": "POSITIVE", "score": 0.5}]],
    )
    assert detect_framework(model) == "huggingface"


def test_huggingface_resolver_basic():
    model = _make_text_classification_pipeline(
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        outputs=[
            [{"label": "POSITIVE", "score": 0.9}, {"label": "NEGATIVE", "score": 0.1}],
            [{"label": "NEGATIVE", "score": 0.7}, {"label": "POSITIVE", "score": 0.3}],
        ],
    )
    X = ["great movie", "bad movie"]

    resolver = get_resolver(model)
    bundle = resolver(model, X)

    assert bundle.framework == "huggingface"
    assert bundle.y_pred.shape == (2,)
    assert bundle.y_prob.shape == (2, 2)
    assert bundle.metadata["resolver"] == "huggingface"
    assert list(bundle.class_labels) == ["NEGATIVE", "POSITIVE"]
    assert list(bundle.y_pred) == ["POSITIVE", "NEGATIVE"]


def test_huggingface_integration_analyze():
    id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    model = _make_text_classification_pipeline(
        id2label=id2label,
        outputs=[
            [
                {"label": "POSITIVE", "score": 0.7},
                {"label": "NEUTRAL", "score": 0.2},
                {"label": "NEGATIVE", "score": 0.1},
            ],
            [
                {"label": "NEGATIVE", "score": 0.6},
                {"label": "NEUTRAL", "score": 0.3},
                {"label": "POSITIVE", "score": 0.1},
            ],
            [
                {"label": "NEUTRAL", "score": 0.5},
                {"label": "POSITIVE", "score": 0.3},
                {"label": "NEGATIVE", "score": 0.2},
            ],
            [
                {"label": "NEGATIVE", "score": 0.5},
                {"label": "POSITIVE", "score": 0.3},
                {"label": "NEUTRAL", "score": 0.2},
            ],
        ],
    )
    X = ["great movie", "terrible movie", "it was fine", "not for me"]
    y = np.array(["POSITIVE", "NEGATIVE", "NEUTRAL", "NEGATIVE"])

    report = analyze(model, X, y, verbose=False)

    assert isinstance(report, TrustReport)
    assert report.metadata["framework"] == "huggingface"
    assert "huggingface" in report.metadata["backend"]["resolver"]


def test_huggingface_manual_override():
    id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    model = _make_text_classification_pipeline(
        id2label=id2label,
        outputs=[
            [
                {"label": "POSITIVE", "score": 0.7},
                {"label": "NEUTRAL", "score": 0.2},
                {"label": "NEGATIVE", "score": 0.1},
            ],
            [
                {"label": "NEGATIVE", "score": 0.6},
                {"label": "NEUTRAL", "score": 0.3},
                {"label": "POSITIVE", "score": 0.1},
            ],
        ],
    )
    X = ["great movie", "terrible movie"]
    y = np.array(["POSITIVE", "NEGATIVE"])

    custom_preds = np.array(["POSITIVE", "NEUTRAL"])
    report = analyze(model, X, y, y_pred=custom_preds, verbose=False)

    assert np.array_equal(report.y_pred, custom_preds)
    assert report.metadata["framework"] == "huggingface"


def test_huggingface_non_classification_pipeline_rejection():
    model = _make_non_classification_pipeline()
    X = ["some long document to summarize"]
    y = np.array([0])

    with pytest.raises(UnsupportedModelError):
        analyze(model, X, y, verbose=False)


def test_huggingface_resolver_no_id2label_fallback():
    model = _make_text_classification_pipeline(
        id2label=None,
        outputs=[
            [{"label": "joy", "score": 0.6}, {"label": "anger", "score": 0.4}],
            [{"label": "anger", "score": 0.2}, {"label": "joy", "score": 0.8}],
            [{"label": "joy", "score": 0.3}, {"label": "anger", "score": 0.7}],
        ],
    )
    X = ["a", "b", "c"]

    resolver = get_resolver(model)
    bundle = resolver(model, X)

    assert bundle.framework == "huggingface"
    assert bundle.y_prob.shape == (3, 2)
    # Column order comes from the first sample: joy=col0, anger=col1.
    assert list(bundle.class_labels) == ["joy", "anger"]
    np.testing.assert_allclose(bundle.y_prob, [[0.6, 0.4], [0.8, 0.2], [0.3, 0.7]])
    assert list(bundle.y_pred) == ["joy", "joy", "anger"]


def test_huggingface_resolver_no_id2label_inconsistent_labels_raises():
    model = _make_text_classification_pipeline(
        id2label=None,
        outputs=[
            [{"label": "joy", "score": 0.6}, {"label": "anger", "score": 0.4}],
            [{"label": "sadness", "score": 0.5}, {"label": "anger", "score": 0.5}],
        ],
    )
    X = ["a", "b"]

    resolver = get_resolver(model)
    with pytest.raises(ValueError, match="not present in the canonical label set"):
        resolver(model, X)


def test_huggingface_resolver_explicit_y_prob_2d():
    class ExplodingPipeline(Exception):
        pass

    def _boom(*_args, **_kwargs):
        raise ExplodingPipeline("model(...) should not be called when y_prob is supplied")

    model = _make_text_classification_pipeline(
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        outputs=[],  # unused; __call__ is overridden below to assert it's never hit
    )
    model.__class__.__call__ = _boom

    explicit_y_prob = np.array([[0.2, 0.8], [0.9, 0.1]])
    resolver = get_resolver(model)
    bundle = resolver(model, ["a", "b"], y_prob=explicit_y_prob)

    assert bundle.framework == "huggingface"
    np.testing.assert_array_equal(bundle.y_prob, explicit_y_prob)
    assert list(bundle.class_labels) == ["NEGATIVE", "POSITIVE"]
    assert list(bundle.y_pred) == ["POSITIVE", "NEGATIVE"]


def test_huggingface_resolver_explicit_y_prob_1d_normalized():
    model = _make_text_classification_pipeline(
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        outputs=[],
    )

    def _boom(*_args, **_kwargs):
        raise AssertionError("model(...) should not be called when y_prob is supplied")

    model.__class__.__call__ = _boom

    explicit_y_prob_1d = np.array([0.8, 0.2])
    resolver = get_resolver(model)
    bundle = resolver(model, ["a", "b"], y_prob=explicit_y_prob_1d)

    assert bundle.y_prob.shape == (2, 2)
    np.testing.assert_allclose(bundle.y_prob, [[0.2, 0.8], [0.8, 0.2]])
    assert list(bundle.y_pred) == ["POSITIVE", "NEGATIVE"]
