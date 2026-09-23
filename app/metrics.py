"""填充总面积与面积质心（metrics）只读分析。

对一份已完成包含层级分类的轮廓列表（:func:`app.svggeo.analyze_contours`
的产物），按层级交替计入实际填充区域：

- 外环（outer，深度偶数）：面积计入；
- 孔洞（hole，深度奇数）：面积扣减；孔中再嵌套的“岛”深度为偶数，
  重新计入，依此类推；
- 开放轮廓（未以 ``Z`` 闭合）与退化轮廓（面积为 0 或顶点不足）不参与。

面积与质心均按现有曲线细分（贝塞尔/椭圆弧展开为折线）后的多边形
近似计算：每个环用鞋带公式求面积、用多边形一阶矩求质心
（质心公式对绘制方向不敏感），总质心为各环**按计入面积加权**的
加权平均（孔洞以负权重参与）。

整个计算只读、即时完成：不依赖已保存的规范化结果，也不写库。
"""

from __future__ import annotations

from typing import Any

from .svggeo import Contour

# 输出坐标/面积的量化位数（仅影响 JSON 展示，不影响判定）
_METRICS_DECIMALS = 9
# “总面积被孔/岛完全抵消”判定的相对容差：吸收细分与浮点误差
_ZERO_AREA_REL_TOL = 1e-9


def polygon_area_centroid(
    points: list[tuple[float, float]]
) -> tuple[float, tuple[float, float] | None]:
    """细分多边形的（有符号面积，质心）。

    鞋带公式求有符号面积，一阶矩公式求质心；分子与面积同时带号，
    比值与绘制方向无关，因此质心对顺/逆时针轮廓一致。

    顶点不足或面积恒为 0 时返回 ``(0.0, None)``。
    """
    n = len(points)
    if n < 3:
        return 0.0, None

    twice_area = 0.0
    mx = my = 0.0
    for k in range(n):
        x1, y1 = points[k]
        x2, y2 = points[(k + 1) % n]
        cross = x1 * y2 - x2 * y1
        twice_area += cross
        mx += (x1 + x2) * cross
        my += (y1 + y2) * cross

    if twice_area == 0.0:
        return 0.0, None
    area = twice_area / 2.0
    cx = mx / (3.0 * twice_area)
    cy = my / (3.0 * twice_area)
    return area, (cx, cy)


def compute_metrics(contours: list[Contour]) -> dict[str, Any]:
    """计算填充总面积与面积质心。

    参数:
        contours: :func:`app.svggeo.analyze_contours` 返回的已分类轮廓

    返回:
        ``{area, centroid, contributing_contour_count, contours,
           excluded_contours}``；无有效闭合区域（或总面积被孔/岛
        完全抵消）时 ``area=0``、``centroid=None``。
    """
    ordered = sorted(contours, key=lambda c: c.original_index)

    details: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    total_area = 0.0          # 外环 +、孔洞 - 的计入面积
    moment_x = moment_y = 0.0
    max_ring_area = 0.0       # 参与环的最大（无符号）面积，用于零面积容差

    for c in ordered:
        if c.role not in ("outer", "hole"):
            excluded.append({
                "original_index": c.original_index,
                "role": c.role,
                "reason": "open" if c.role == "open" else "degenerate",
            })
            continue

        signed_area, ring_centroid = polygon_area_centroid(c.points)
        ring_area = abs(signed_area)
        sign = 1.0 if c.role == "outer" else -1.0

        total_area += sign * ring_area
        if ring_centroid is not None:
            moment_x += sign * ring_area * ring_centroid[0]
            moment_y += sign * ring_area * ring_centroid[1]
        max_ring_area = max(max_ring_area, ring_area)

        details.append({
            "original_index": c.original_index,
            "role": c.role,
            "depth": c.depth,
            "area": _round(ring_area),
            "centroid": None if ring_centroid is None
            else {"x": _round(ring_centroid[0]),
                  "y": _round(ring_centroid[1])},
        })

    # 严格包含层级下孔面积恒小于父外环，总面积应为正；
    # tol 仅用于吸收浮点误差及理论上的完全抵消（抵消时质心无定义）
    tol = _ZERO_AREA_REL_TOL * max(1.0, max_ring_area)
    if not details or total_area <= tol:
        centroid = None
        area = 0.0
    else:
        centroid = {"x": _round(moment_x / total_area),
                    "y": _round(moment_y / total_area)}
        area = _round(total_area)

    return {
        "area": area,
        "centroid": centroid,
        "contributing_contour_count": len(details),
        "contours": details,
        "excluded_contours": excluded,
    }


def _round(v: float) -> float:
    """量化展示值，并把 -0.0 归一为 0.0。"""
    r = round(v, _METRICS_DECIMALS)
    return r + 0.0 if isinstance(r, float) else r
