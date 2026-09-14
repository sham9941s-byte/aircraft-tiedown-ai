from typing import Any


def _bbox_from_points(
    p1: tuple[int, int],
    p2: tuple[int, int],
    padding: int = 8,
) -> list[int]:
    x1, y1 = p1
    x2, y2 = p2

    return [
        min(x1, x2) - padding,
        min(y1, y2) - padding,
        max(x1, x2) + padding,
        max(y1, y2) + padding,
    ]


def _clip_bbox(
    bbox: list[int],
    width: int,
    height: int,
) -> list[int]:
    x1, y1, x2, y2 = bbox

    return [
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    ]


def build_annotation(
    assessment_id,
    image_id,
    view,
    scenario,
    engine,
    truck,
    tiedown,
    suspension,
    width,
    height,
) -> dict[str, Any]:

    objects = [
        {
            "class": "engine",
            "bbox": [
                engine.x,
                engine.y,
                engine.x + engine.width,
                engine.y + engine.height,
            ],
            "visibility": "visible",
            "occlusion": scenario.occlusion,
        },
        {
            "class": "trailer",
            "bbox": [
                truck.bed_x,
                truck.bed_y,
                truck.bed_x + truck.bed_width,
                truck.bed_y + truck.bed_height,
            ],
            "visibility": "visible",
            "occlusion": 0.0,
        },
        {
            "class": "pneumatic_suspension",
            "bbox": [
                suspension.x,
                suspension.y,
                suspension.x + suspension.width,
                suspension.y + suspension.height,
            ],
            "visibility": "visible",
            "occlusion": 0.0,
            "quality": scenario.suspension_state,
        },
    ]

    # The renderer uses the same truck-bed anchor positions.
    for side, anchors in [
        ("left", tiedown.left_anchors),
        ("right", tiedown.right_anchors),
    ]:
        target_x = (
            truck.bed_x + 30
            if side == "left"
            else truck.bed_x + truck.bed_width - 30
        )

        for idx, (ax, ay) in enumerate(anchors):
            target_y = (
                truck.bed_y
                + truck.bed_height // 2
                + (idx - 1) * 12
            )

            # Match the renderer's loose-strap modification.
            if scenario.strap_state == "loose" and idx == 0:
                target_y -= 45

            # Match the renderer's unconnected-strap modification.
            render_ax = ax

            if scenario.strap_state == "unconnected" and idx == 0:
                render_ax = ax + (55 if side == "left" else -55)

            bbox = _bbox_from_points(
                (render_ax, ay),
                (target_x, target_y),
                padding=10,
            )

            objects.append(
                {
                    "class": "strap",
                    "bbox": _clip_bbox(bbox, width, height),
                    "side": side,
                    "index": idx,
                    "quality": scenario.strap_state,
                }
            )

    return {
        "assessment_id": assessment_id,
        "image_id": image_id,
        "view": view,
        "width": width,
        "height": height,
        "scenario": scenario.name,
        "expected_decision": scenario.expected_decision,
        "objects": objects,
    }
