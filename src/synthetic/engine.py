from dataclasses import dataclass

@dataclass
class Engine:
    x: int
    y: int
    width: int
    height: int

def make_engine(width: int, height: int) -> Engine:
    ew, eh = int(width * 0.34), int(height * 0.34)
    return Engine(
        x=int(width * 0.33),
        y=int(height * 0.30),
        width=ew,
        height=eh,
    )
