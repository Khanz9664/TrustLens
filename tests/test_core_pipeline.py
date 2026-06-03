import numpy as np
import pytest

from trustlens.core.pipeline import _encode_labels_for_probability_columns


def test_encode_labels_rejects_class_label_length_mismatch():
    y_true = np.array(["mouse", "cat"])
    class_labels = np.array(["mouse", "cat"])

    with pytest.raises(
        ValueError, match=r"class_labels length \(2\).*probability column shape \(3 columns\)"
    ):
        _encode_labels_for_probability_columns(y_true, 3, class_labels)
