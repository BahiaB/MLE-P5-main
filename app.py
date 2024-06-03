from fastapi import FastAPI # type: ignore
from joblib import load # type: ignore
from pydantic import BaseModel # type: ignore
from final_model import preprocess_data,final_model, vectorize_data, select_top_n_tags

app = FastAPI()

try:
    preprocessing = load('preprocess_function.joblib')
    vectorizer = load('vectorize_function.joblib')
    model = load('final_model.joblib')
except Exception as e:
    print(f"Failed to load models: {e}")
    raise e

@app.get("/")
def read_root():
    return {"Hello": "World"}
class Item(BaseModel):
    text: str

@app.post("/predict/")
async def make_prediction(item: Item):

    # Prétrait
    processed_data = preprocessing([item.text])
    vec_data = vectorizer(processed_data)
    prediction = model(vec_data)
    
    return {"prediction": prediction}
