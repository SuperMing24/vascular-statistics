import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from vascular_statistics.extract_metadata import parse_sample_path


def test_parse_huaien_directory_timing():
    parsed = parse_sample_path(
        "/data/BCAS_1st/20241009_A192_D0/angiogram/angiogram_crop_97_111.mat",
        "/data",
    )

    assert parsed["group"] == "BCAS_1st"
    assert parsed["batch_id"] == "A192"
    assert parsed["daypoint"] == "D0"
    assert parsed["study_day"] == 0
    assert parsed["study_day_label"] == "D0"
    assert parsed["acquisition_date"] == "2024-10-09"
    assert parsed["timepoint_id"] == "20241009_D0"
    assert parsed["source_time_token"] == "20241009_A192_D0"
    assert parsed["sample_key"] == "BCAS_1st/20241009_A192_D0/angiogram_crop_97_111"


def test_parse_xiaoqian_filename_timing():
    parsed = parse_sample_path(
        (
            "/data/ACTH_angiogram/ACTH_HNK/"
            "20251222_A47_D15_angiogram_crop_100_165.mat"
        ),
        "/data",
    )

    assert parsed["group"] == "ACTH_angiogram"
    assert parsed["batch_id"] == "A47"
    assert parsed["daypoint"] == "D15"
    assert parsed["study_day"] == 15
    assert parsed["study_day_label"] == "D15"
    assert parsed["acquisition_date"] == "2025-12-22"
    assert parsed["timepoint_id"] == "20251222_D15"
    assert parsed["source_time_token"] == "20251222_A47_D15"
    assert (
        parsed["sample_key"]
        == "ACTH_angiogram/ACTH_HNK/20251222_A47_D15_angiogram_crop_100_165"
    )
