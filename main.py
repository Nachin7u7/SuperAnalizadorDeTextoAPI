from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import pandas as pd
from transformers import pipeline
import spacy
import openai

app = FastAPI()


class SentimentInput(BaseModel):
    text: str


class AnalysisInput(BaseModel):
    text: str


models_info = {
    "sentiment": {"name": "Sentiment Analysis Model", "type": "NLP"},
    "nlp": {"name": "Spacy NLP Model", "type": "NLP"},
}

inference_reports = []


@app.get("/status")
def read_status():
    return {"service": "Running", "models": models_info}


@app.post("/sentiment")
def analyze_sentiment(input_data: SentimentInput):
    start_time = time.time()

    sentiment_analyzer = pipeline(
        'sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

    result = sentiment_analyzer(input_data.text)
    sentiment_value = result[0]['score']

    execution_time = time.time() - start_time

    response_data = {
        "sentiment_value": sentiment_value,
        "execution_time": execution_time,
        "model_info": {"name": "Sentiment Analysis Model", "type": "NLP"},
    }

    inference_reports.append(response_data)

    return response_data


nlp_model = spacy.load("en_core_web_sm")


@app.post("/analysis")
def analyze_text(input_data: AnalysisInput):
    start_time = time.time()

    doc = nlp_model(input_data.text)
    pos_tags = [token.pos_ for token in doc]
    ner_entities = [{"text": ent.text, "label": ent.label_}
                    for ent in doc.ents]

    sentiment_analyzer = pipeline(
        'sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    sentiment_result = sentiment_analyzer(input_data.text)
    sentiment_value = sentiment_result[0]['score']

    sentiment_category = "Positive" if sentiment_value > 0.7 else "Negative"
    embedding = "Embedding data"

    execution_time = time.time() - start_time

    response_data = {
        "pos_tags": pos_tags,
        "ner_entities": ner_entities,
        "sentiment_value": sentiment_value,
        "sentiment_category": sentiment_category,
        "embedding": embedding,
        "execution_time": execution_time,
        "model_info": {"name": "Combined Analysis Model", "type": "NLP"},
    }

    inference_reports.append(response_data)

    return response_data


@app.get("/reports")
def generate_reports():

    df = pd.DataFrame(inference_reports)

    csv_data = df.to_csv(index=False)

    return {"csv_data": csv_data}


class AnalysisV2Input(BaseModel):
    text: str
    prompt: str


openai.api_key = "KEYYYYYY" ## NOTA PARA MI MISMO --- > quitar esto para subirlo


class AnalysisV2Input(BaseModel):
    text: str
    prompt: str


@app.post("/analysis_v2")
def analyze_text_v2(input_data: AnalysisV2Input):
    start_time = time.time()

    # POS tagging y NER
    doc = nlp_model(input_data.text)
    pos_tags = [token.pos_ for token in doc]
    ner_entities = [{"text": ent.text, "label": ent.label_}
                    for ent in doc.ents]

    sentiment_prompt = 0.8

    lm_response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=input_data.text,
        max_tokens=150,
    )
    lm_output = lm_response["choices"][0]["text"]

    execution_time = time.time() - start_time

    response_data = {
        "pos_tags": pos_tags,
        "ner_entities": ner_entities,
        "sentiment_prompt": sentiment_prompt,
        "lm_output": lm_output,
        "execution_time": execution_time,
        "model_info": {"name": "Combined Analysis Model", "type": "NLP"},
    }

    inference_reports.append(response_data)

    return response_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
