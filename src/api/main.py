from typing import Annotated
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from src.training.api import router as training_router


from src.inference.pipeline import analyze_images


app = FastAPI(
    title="Vision Wings - Aircraft Engine Tie-Down Detector"
)

app.include_router(training_router)

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

        path = f"data/tmp_{index}_{image.filename}"

        with open(path, "wb") as f:
            f.write(await image.read())

        temp_paths.append(path)

    result = analyze_images(temp_paths)

    return result


# IMPORTANT:
# Keep this AFTER all API routes.
app.mount(
    "/",
    StaticFiles(directory="static", html=True),
    name="static",
)