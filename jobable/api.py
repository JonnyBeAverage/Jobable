from fastapi import FastAPI
from jobable.ml_logic.cover_letter import create_cover_letter

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Jobable model API running"}

@app.post("/generate")
def generate(data: dict):

    letter = create_cover_letter(
        data["resume"],
        data["job_description"]
    )

    return {"cover_letter": letter}
