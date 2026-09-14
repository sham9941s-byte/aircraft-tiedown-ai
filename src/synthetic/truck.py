from dataclasses import dataclass

@dataclass
class Truck:
    bed_x: int
    bed_y: int
    bed_width: int
    bed_height: int
    suspension_x: int
    suspension_y: int

def make_truck(width: int, height: int) -> Truck:
    return Truck(
        bed_x=int(width * 0.12),
        bed_y=int(height * 0.60),
        bed_width=int(width * 0.76),
        bed_height=int(height * 0.16),
        suspension_x=int(width * 0.73),
        suspension_y=int(height * 0.76),
    )
