from dataclasses import dataclass
from typing import Literal

ScenarioName = Literal[
    "correct_three_straps",
    "correct_two_straps",
    "loose_strap",
    "unconnected_strap",
    "missing_connection_point",
    "bad_suspension",
    "occluded_tiedown",
    "poor_image",
]

@dataclass(frozen=True)
class Scenario:
    name: ScenarioName
    straps_per_side: int
    strap_state: str
    suspension_state: str
    occlusion: float = 0.0
    blur: float = 0.0
    expected_decision: str = "MORE_IMAGES_REQUIRED"

SCENARIOS = [
    Scenario("correct_three_straps", 3, "good", "good", expected_decision="GOOD_TO_GO"),
    Scenario("correct_two_straps", 2, "good", "good", expected_decision="GOOD_TO_GO"),
    Scenario("loose_strap", 2, "loose", "good", expected_decision="TIE_DOWN_INCORRECT"),
    Scenario("unconnected_strap", 2, "unconnected", "good", expected_decision="TIE_DOWN_INCORRECT"),
    Scenario("missing_connection_point", 2, "missing_anchor", "good", expected_decision="TIE_DOWN_INCORRECT"),
    Scenario("bad_suspension", 2, "good", "bad", expected_decision="TIE_DOWN_INCORRECT"),
    Scenario("occluded_tiedown", 2, "occluded", "good", occlusion=0.55),
    Scenario("poor_image", 2, "good", "good", blur=1.0),
]

def get_scenario(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise ValueError(f"Unknown scenario: {name}")
