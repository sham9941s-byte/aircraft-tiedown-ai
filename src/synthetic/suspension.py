from dataclasses import dataclass

@dataclass
class SuspensionGeometry:
    x: int
    y: int
    width: int
    height: int

def make_suspension(truck) -> SuspensionGeometry:
    return SuspensionGeometry(
        x=truck.suspension_x,
        y=truck.suspension_y,
        width=110,
        height=70,
    )
