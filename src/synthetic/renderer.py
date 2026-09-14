from pathlib import Path

import cv2
import numpy as np


def _poly(img, pts, value):
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], value)


def render(width, height, engine, truck, tiedown, suspension, scenario, view, rng):
    img = np.full((height, width, 3), 235, dtype=np.uint8)

    # Ground / trailer
    cv2.rectangle(
        img,
        (0, int(height * 0.78)),
        (width, height),
        (80, 80, 80),
        -1,
    )

    cv2.rectangle(
        img,
        (truck.bed_x, truck.bed_y),
        (
            truck.bed_x + truck.bed_width,
            truck.bed_y + truck.bed_height,
        ),
        (150, 150, 150),
        -1,
    )

    # Engine body
    ex, ey = engine.x, engine.y
    ew, eh = engine.width, engine.height

    _poly(
        img,
        [
            (ex, ey + eh // 2),
            (ex + ew // 5, ey),
            (ex + 4 * ew // 5, ey),
            (ex + ew, ey + eh // 2),
            (ex + 4 * ew // 5, ey + eh),
            (ex + ew // 5, ey + eh),
        ],
        (125, 125, 125),
    )

    # Engine core
    cv2.ellipse(
        img,
        (ex + ew // 2, ey + eh // 2),
        (ew // 4, eh // 3),
        0,
        0,
        360,
        (185, 185, 185),
        -1,
    )

    # Tie-down straps
    for side, anchors in [
        ("left", tiedown.left_anchors),
        ("right", tiedown.right_anchors),
    ]:

        for idx, (ax, ay) in enumerate(anchors):

            target_x = (
                truck.bed_x + 30
                if side == "left"
                else truck.bed_x + truck.bed_width - 30
            )

            target_y = (
                truck.bed_y
                + truck.bed_height // 2
                + (idx - 1) * 12
            )

            # Loose strap
            if scenario.strap_state == "loose" and idx == 0:
                target_y -= 45

            # Unconnected strap
            render_ax = ax

            if scenario.strap_state == "unconnected" and idx == 0:
                render_ax = ax + (
                    55 if side == "left" else -55
                )

            cv2.line(
                img,
                (render_ax, ay),
                (target_x, target_y),
                (45, 45, 45),
                10,
                cv2.LINE_AA,
            )

            # Anchor point
            if not (
                scenario.strap_state == "missing_anchor"
                and idx == 0
            ):
                cv2.circle(
                    img,
                    (target_x, target_y),
                    12,
                    (30, 30, 30),
                    -1,
                )

    # Suspension
    sx, sy = suspension.x, suspension.y
    sw, sh = suspension.width, suspension.height

    cv2.rectangle(
        img,
        (sx, sy),
        (sx + sw, sy + sh),
        (205, 205, 205),
        -1,
    )

    suspension_value = (
        55
        if scenario.suspension_state == "bad"
        else 170
    )

    cv2.rectangle(
        img,
        (sx + 20, sy + 15),
        (sx + sw - 20, sy + sh - 15),
        (suspension_value,) * 3,
        -1,
    )

    # ---------------------------------------------------------
    # View framing
    # ---------------------------------------------------------

    if view == "left_side":

        crop = img[:, :int(width * 0.65)]
        img = cv2.resize(crop, (width, height))

    elif view == "right_side":

        crop = img[:, int(width * 0.35):]
        img = cv2.resize(crop, (width, height))

    elif view == "pneumatic_suspension":

        crop = img[
            int(height * 0.52):,
            int(width * 0.55):
        ]

        img = cv2.resize(crop, (width, height))

    elif view == "left_closeup":

        # Focus on the left engine-side tie-down connections.
        x1 = int(width * 0.18)
        x2 = int(width * 0.58)
        y1 = int(height * 0.20)
        y2 = int(height * 0.78)

        crop = img[y1:y2, x1:x2]
        img = cv2.resize(crop, (width, height))

    elif view == "right_closeup":

        # Focus on the right engine-side tie-down connections.
        x1 = int(width * 0.42)
        x2 = int(width * 0.82)
        y1 = int(height * 0.20)
        y2 = int(height * 0.78)

        crop = img[y1:y2, x1:x2]
        img = cv2.resize(crop, (width, height))

    # Controlled degradation
    if scenario.blur:
        img = cv2.GaussianBlur(
            img,
            (0, 0),
            5.0,
        )

    if scenario.occlusion:
        ox = int(width * 0.30)
        oy = int(height * 0.20)
        ow = int(width * scenario.occlusion * 0.65)
        oh = int(height * 0.35)

        cv2.rectangle(
            img,
            (ox, oy),
            (ox + ow, oy + oh),
            (235, 235, 235),
            -1,
        )

    # Small sensor noise
    noise = rng.normal(
        0,
        3.0,
        img.shape,
    ).astype(np.int16)

    img = np.clip(
        img.astype(np.int16) + noise,
        0,
        255,
    ).astype(np.uint8)

    return img
