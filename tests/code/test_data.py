import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tagifai import data


class TestLabelEncoder:
    @classmethod
    def setup_class(cls):
        """Called before every class initialization."""
        pass

    @classmethod
    def teardown_class(cls):
        """Calles after every class initialization."""
        pass

    def setup_method(self):
        """Called before every method."""
        self.label_encoder = data.LabelEncoder()

    def teardown_method(self):
        """Called after every method."""
        del self.label_encoder

    def test_empty_init(self):
        label_encoder = data.LabelEncoder()
        assert label_encoder.index_to_class == {}
        assert len(label_encoder.classes) == 0

    def test_dict_init(self):
        class_to_index = {"apple": 0, "banana": 1}
        label_encoder = data.LabelEncoder(class_to_index=class_to_index)
        assert label_encoder.index_to_class == {0: "apple", 1: "banana"}
        assert len(label_encoder.classes) == 2

    def test_len(self):
        assert len(self.label_encoder) == 0

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as dp:
            fp = Path(dp, "label_encoder.json")
            self.label_encoder.save(fp=fp)
            label_encoder = data.LabelEncoder.load(fp=fp)
            assert len(label_encoder.classes) == 0

    def test_str(self):
        assert str(data.LabelEncoder()) == "<LabelEncoder(num_classes=0)>"

    def test_fit(self):
        label_encoder = data.LabelEncoder()
        label_encoder.fit(["apple", "apple", "banana"])
        assert "apple" in label_encoder.class_to_index
        assert "banana" in label_encoder.class_to_index
        assert len(label_encoder.classes) == 2

    def test_encode_decode(self):
        class_to_index = {"apple": 0, "banana": 1}
        y_encoded = [0, 0, 1]
        y_decoded = ["apple", "apple", "banana"]
        label_encoder = data.LabelEncoder(class_to_index=class_to_index)
        label_encoder.fit(["apple", "apple", "banana"])
        assert np.array_equal(label_encoder.encode(y_decoded), np.array(y_encoded))
        assert label_encoder.decode(y_encoded) == y_decoded


@pytest.mark.parametrize(
    "text, lower, stem, stopwords, cleaned_text",
    [
        ("Hello worlds", False, False, [], "Hello worlds"),
        ("Hello worlds", True, False, [], "hello worlds"),
    ],
)
def test_preprocess(text, lower, stem, stopwords, cleaned_text):
    assert data.clean_text(text=text, lower=lower, stem=stem, stopwords=stopwords) == cleaned_text


@pytest.fixture(scope="module")
def df():
    data = [
        {"title": "a0", "description": "b0", "tag": "c0"},
        {"title": "a1", "description": "b1", "tag": "c1"},
        {"title": "a2", "description": "b2", "tag": "c1"},
        {"title": "a3", "description": "b3", "tag": "c2"},
        {"title": "a4", "description": "b4", "tag": "c2"},
        {"title": "a5", "description": "b5", "tag": "c2"},
    ]
    df = pd.DataFrame(data * 10)
    return df


@pytest.mark.parametrize(
    "labels, unique_labels",
    [
        ([], ["other"]),  # no set of approved labels
        (["c3"], ["other"]),  # no overlap b/w approved/actual labels
        (["c0"], ["c0", "other"]),  # partial overlap
        (["c0", "c1", "c2"], ["c0", "c1", "c2"]),  # complete overlap
    ],
)
def test_replace_oos_labels(df, labels, unique_labels):
    replaced_df = data.replace_oos_labels(
        df=df.copy(), labels=labels, label_col="tag", oos_label="other"
    )
    assert set(replaced_df.tag.unique()) == set(unique_labels)
