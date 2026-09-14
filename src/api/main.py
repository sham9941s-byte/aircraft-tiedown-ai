from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.training.api import router as training_router
from src.inference.pipeline import analyze_images


BASE_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

app = FastAPI(
    title="Vision Wings - Aircraft Engine Tie-Down Detector"
)

app.include_router(training_router)


@app.get("/")
def inspection_page():
    return FileResponse(STATIC_DIR / "inspection.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {
        "service": "vision-wings-aircraft-tiedown",
        "version": "0.2.0",
    }


@app.post("/assess")
async def assess(
    images: Annotated[list[UploadFile], File(...)]
):
    if len(images) < 6:
        raise HTTPException(
            status_code=400,
            detail="Minimum 6 images are required.",
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    temp_paths = []

    for index, image in enumerate(images, start=1):
        if (
            not image.content_type
            or not image.content_type.startswith("image/")
        ):
            raise HTTPException(
                status_code=400,
                detail=f"File {image.filename} is not an image.",
            )

        filename = Path(image.filename or f"image_{index}.jpg").name
        path = DATA_DIR / f"tmp_{index}_{filename}"

        with open(path, "wb") as f:
            f.write(await image.read())

        temp_paths.append(str(path))

    result = analyze_images(temp_paths)

    return result


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)