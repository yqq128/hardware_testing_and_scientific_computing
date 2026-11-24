import pandas as pd

from aoi.data import build_clean, first_pass, with_reference


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "inspection_id": ["i1", "i2", "i2", "i3", "i4"],
            "product_id": ["p1", "p1", "p1", "p2", "p3"],
            "inspection_time": pd.to_datetime(
                [
                    "2026-10-01 23:50",
                    "2026-10-02 00:10",  # p1 的复测，跨过午夜
                    "2026-10-02 00:10",  # i2 的重复行（脏数据）
                    "2026-10-02 08:00",
                    "2026-10-02 09:00",
                ]
            ),
            "aoi_result": ["NG", "OK", "OK", "OK", "NG"],
            "reference_result": ["OK", "OK", "OK", "NG", None],
        }
    )


def test_duplicate_inspection_ids_are_dropped():
    clean = build_clean(_raw())
    assert clean["inspection_id"].tolist() == ["i1", "i2", "i3", "i4"]


def test_first_pass_is_the_earliest_inspection_even_across_midnight():
    clean = build_clean(_raw())
    first = first_pass(clean)
    assert first["inspection_id"].tolist() == ["i1", "i3", "i4"]
    assert clean.set_index("inspection_id").loc["i2", "inspection_seq"] == 2


def test_confusion_flags_follow_reference_result():
    clean = build_clean(_raw()).set_index("inspection_id")
    assert clean.loc["i1", "is_false_positive"]  # AOI 报警，参考为良品
    assert clean.loc["i3", "is_miss"]  # AOI 放行，参考为缺陷
    assert not clean.loc["i4", "has_reference"]  # 没有参考判定，不能算误报或漏检
    assert not clean.loc["i4", "is_false_positive"]


def test_with_reference_filters_unlabelled_rows():
    clean = build_clean(_raw())
    assert "i4" not in with_reference(clean)["inspection_id"].tolist()
