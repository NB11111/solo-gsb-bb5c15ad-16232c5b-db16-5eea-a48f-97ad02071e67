"""填充总面积/面积质心计算的单元测试。"""

import math

from app.metrics import compute_metrics, polygon_area_centroid
from app.svggeo import analyze_contours


def _metrics(d: str, fill_rule: str = "nonzero") -> dict:
    contours, _ = analyze_contours(d, fill_rule)
    return compute_metrics(contours)


def test_single_outer_square():
    m = _metrics("M0,0h10v10h-10Z")
    assert m["area"] == 100.0
    assert m["centroid"] == {"x": 5.0, "y": 5.0}
    assert m["contributing_contour_count"] == 1
    assert m["excluded_contours"] == []
    assert m["contours"][0]["role"] == "outer"
    assert m["contours"][0]["depth"] == 0
    assert m["contours"][0]["area"] == 100.0
    assert m["contours"][0]["centroid"] == {"x": 5.0, "y": 5.0}


def test_outer_with_hole_subtracts_area():
    # 10x10 外环减 4x4 中心孔 -> 84；填充关于中心对称 -> 质心 (5,5)
    m = _metrics("M0,0h10v10h-10Z M3,3h4v4h-4Z")
    assert m["area"] == 84.0
    assert m["centroid"] == {"x": 5.0, "y": 5.0}
    assert m["contributing_contour_count"] == 2
    roles = {c["original_index"]: c["role"] for c in m["contours"]}
    assert roles == {0: "outer", 1: "hole"}


def test_island_inside_hole_is_added_back():
    # 外环 0..10（100）- 孔 2..8（36）+ 岛 4..6（4）= 68
    m = _metrics(
        "M0,0h10v10h-10Z M2,2h6v6h-6Z M4,4h2v2h-2Z")
    assert m["area"] == 68.0
    # 图形关于 (5,5) 对称
    assert m["centroid"] == {"x": 5.0, "y": 5.0}
    roles = [c["role"] for c in m["contours"]]
    assert roles == ["outer", "hole", "outer"]
    depths = [c["depth"] for c in m["contours"]]
    assert depths == [0, 1, 2]


def test_off_center_hole_shifts_centroid_toward_opposite_corner():
    # 外环 0..10（质心 5,5，面积100），孔 0..4 在左下（质心 2,2，面积16）
    m = _metrics("M0,0h10v10h-10Z M0,0h4v4h-4Z")
    assert m["area"] == 84.0
    cx = (100 * 5 - 16 * 2) / 84
    cy = (100 * 5 - 16 * 2) / 84
    assert math.isclose(m["centroid"]["x"], cx, abs_tol=1e-9)
    assert math.isclose(m["centroid"]["y"], cy, abs_tol=1e-9)
    assert m["centroid"]["x"] > 5.0  # 填充质心远离孔洞


def test_winding_direction_does_not_affect_area_and_centroid():
    # 外环逆时针书写、孔顺时针书写：层级与计入方式仍由包含关系决定
    m = _metrics("M0,10h10v-10h-10Z M3,7h4v-4h-4Z")
    assert m["area"] == 84.0
    assert m["centroid"] == {"x": 5.0, "y": 5.0}


def test_evenodd_uses_same_depth_alternation():
    m = _metrics(
        "M0,0h10v10h-10Z M2,2h6v6h-6Z M4,4h2v2h-2Z",
        fill_rule="evenodd")
    assert m["area"] == 68.0
    assert m["centroid"] == {"x": 5.0, "y": 5.0}


def test_open_and_degenerate_contours_excluded():
    # 开放折线（>=3 点但未闭合）+ 零面积退化环 + 一个有效方
    d = "M0,0h10v10h-10Z M20,0 L30,5 L20,10 M2,2 L8,2 L2,2 Z"
    m = _metrics(d)
    assert m["area"] == 100.0
    assert m["centroid"] == {"x": 5.0, "y": 5.0}
    assert m["contributing_contour_count"] == 1
    excluded = {(e["original_index"], e["reason"])
                for e in m["excluded_contours"]}
    assert excluded == {(1, "open"), (2, "degenerate")}


def test_only_open_and_degenerate_yields_zero_and_null_centroid():
    d = "M0,0 L10,0 L5,3 M5,5 L6,5 L5,5 Z"
    m = _metrics(d)
    assert m["area"] == 0
    assert m["centroid"] is None
    assert m["contributing_contour_count"] == 0
    assert len(m["excluded_contours"]) == 2


def test_empty_path_yields_zero_and_null_centroid():
    m = _metrics("")
    assert m["area"] == 0
    assert m["centroid"] is None
    assert m["contours"] == []
    assert m["excluded_contours"] == []


def test_curve_flatten_approximates_circle_area():
    # 单位圆：M1,0 + 两段半圆弧构成闭合圆；弧线按现有方案细分为
    # 内接多边形（半弧 16 段 -> 全圆 32 边），面积略小于 π
    m = _metrics("M1,0 A1,1 0 0,1 -1,0 A1,1 0 0,1 1,0 Z")
    assert m["contributing_contour_count"] == 1
    inscribed = 16 * math.sin(math.pi / 16)  # 内接 32 边形面积
    assert math.isclose(m["area"], inscribed, rel_tol=1e-9)
    assert math.isclose(m["area"], math.pi, rel_tol=1e-2)
    assert math.isclose(m["centroid"]["x"], 0.0, abs_tol=1e-9)
    assert math.isclose(m["centroid"]["y"], 0.0, abs_tol=1e-9)


def test_bezier_outer_is_measured_on_flattened_polygon():
    # 控制点共线的二次贝塞尔退化为直边：细分多边形即 10x10 矩形
    m = _metrics("M0,0 Q5,0 10,0 L10,10 L0,10 Z")
    assert math.isclose(m["area"], 100.0, rel_tol=1e-9)
    assert m["centroid"] == {"x": 5.0, "y": 5.0}

    # 真正凸起的二次贝塞尔 P0=(0,0) P1=(10,10) P2=(0,10) + 闭合弦：
    # 解析面积 100/3，质心 (2, 6)；12 段弦细分的误差约 0.7%
    m2 = _metrics("M0,0 Q10,10 0,10 Z")
    assert math.isclose(m2["area"], 100.0 / 3.0, rel_tol=1e-2)
    assert math.isclose(m2["centroid"]["x"], 2.0, abs_tol=0.02)
    assert math.isclose(m2["centroid"]["y"], 6.0, abs_tol=0.02)


def test_two_separate_outers_combined_area_and_centroid():
    # 两个不相交的 10x10 方：面积 200，质心为两者中点 (10,5)
    m = _metrics("M0,0h10v10h-10Z M10,0h10v10h-10Z")
    assert m["area"] == 200.0
    assert m["centroid"] == {"x": 10.0, "y": 5.0}


def test_polygon_centroid_helper_degenerate():
    assert polygon_area_centroid([]) == (0.0, None)
    assert polygon_area_centroid([(0, 0), (1, 1)]) == (0.0, None)
    # 共线三点，twice_area == 0
    assert polygon_area_centroid([(0, 0), (1, 1), (2, 2)])[1] is None
    area, c = polygon_area_centroid([(0, 0), (0, 10), (10, 10), (10, 0)])
    # 逆时针（y 向下）有符号面积为负，但质心与方向无关
    assert area == -100.0
    assert c == (5.0, 5.0)
