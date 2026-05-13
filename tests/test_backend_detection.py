import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from trustlens.backends.registry import detect_framework, get_resolver
from trustlens.backends.types import UnsupportedModelError

class MockModel:
    pass

class SklearnLikeMock:
    def predict(self, X): pass
    def predict_proba(self, X): pass

def test_explicit_override():
    assert detect_framework(MockModel(), framework="sklearn") == "sklearn"
    assert detect_framework(MockModel(), framework="pytorch") == "pytorch"
    assert detect_framework(MockModel(), framework="torch") == "pytorch"  # mapped
    
    with pytest.raises(UnsupportedModelError):
        detect_framework(MockModel(), framework="banana")

def test_module_name_detection():
    # Real sklearn models
    assert detect_framework(RandomForestClassifier()) == "sklearn"
    assert detect_framework(LogisticRegression()) == "sklearn"
    
    # Prefix matching
    class FakeTorchModel:
        __module__ = "torch.nn.modules.conv"
    
    assert detect_framework(FakeTorchModel()) == "pytorch"

def test_capability_fallback():
    assert detect_framework(SklearnLikeMock()) == "sklearn"

def test_unsupported_model():
    with pytest.raises(UnsupportedModelError) as excinfo:
        detect_framework(MockModel())
    assert "Unsupported model type" in str(excinfo.value)
    assert "MockModel" in str(excinfo.value)
    assert "sklearn" in str(excinfo.value)

def test_get_resolver_success():
    resolver = get_resolver(RandomForestClassifier())
    assert callable(resolver)
    # Check if it's the sklearn resolver by name
    assert resolver.__name__ == "resolve"
    assert "sklearn" in resolver.__module__

def test_get_resolver_unsupported():
    with pytest.raises(UnsupportedModelError):
        get_resolver(MockModel())
