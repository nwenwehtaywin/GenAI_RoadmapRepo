from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline


app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Emotion Classifier API is up and running"}

# Endpoint for emotion classification
@app.post("/classify-emotion")
async def classify_emotion(input_data: TextInput):
    pipe = pipeline("text-classification", model="nnhwin/emo_model")
    user_input = input_data.text
    if not user_input:
        raise HTTPException(status_code=400, detail="Please provide some text to classify")

    # Get prediction from the model
    output = pipe(user_input)

    # Format response
    results = [{"label": result["label"], "confidence": result["score"]} for result in output]
    return {"classification_result": results}
