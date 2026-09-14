from dataclasses import dataclass

@dataclass
class TieDownGeometry:
    left_anchors: list[tuple[int, int]]
    right_anchors: list[tuple[int, int]]

def make_tiedown(engine, truck, straps_per_side: int) -> TieDownGeometry:
    n = max(2, min(3, straps_per_side))
    ys = [
        int(engine.y + engine.height * (0.30 + i * 0.30 / max(1, n - 1)))
        for i in range(n)
    ]
    left = [(int(engine.x), y) for y in ys]
    right = [(int(engine.x + engine.width), y) for y in ys]
    return TieDownGeometry(left, right)
