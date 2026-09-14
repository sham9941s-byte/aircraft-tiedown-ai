from dataclasses import dataclass
import cv2


@dataclass
class ImageQuality:
    usable: bool
    width: int
    height: int
    blur_score: float
    brightness: float
    issues: list[str]


def assess_image_quality(image_path: str) -> ImageQuality:
    image = cv2.imread(image_path)

    if image is None:
        return ImageQuality(
            usable=False,
            width=0,
            height=0,
            blur_score=0.0,
            brightness=0.0,
            issues=["Image could not be decoded."],
        )

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Variance of Laplacian is a simple sharpness/blur indicator.
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    brightness = float(gray.mean())

    issues = []

    if width < 640 or height < 480:
        issues.append("Image resolution is too low.")

    if blur_score < 40:
        issues.append("Image appears blurry.")

    if brightness < 30:
        issues.append("Image is too dark.")

    if brightness > 235:
        issues.append("Image is overexposed.")

    return ImageQuality(
        usable=len(issues) == 0,
        width=width,
        height=height,
        blur_score=blur_score,
        brightness=brightness,
        issues=issues,
    )