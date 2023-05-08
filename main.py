from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import MobileBertTokenizer, MobileBertModel
from sklearn.metrics.pairwise import cosine_similarity
import boto3
import os

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data from DynamoDB
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("Campaign-bv5ga2wk7fecbijb73l3zh3t4e-staging")
items = table.scan()["Items"]
data = [
    {
        "id": item["id"],
        "campaignName": item["campaignName"],
        "description": item["description"],
    }
    for item in items
]

# Load MobileBERT tokenizer and model
tokenizer = MobileBertTokenizer.from_pretrained("google/mobilebert-uncased")
model = MobileBertModel.from_pretrained("google/mobilebert-uncased")

# Create MobileBERT embeddings for campaign purposes
purpose_embeddings = [
    model(**tokenizer(description, return_tensors="pt", max_length=512, truncation=True, padding=True)).last_hidden_state[:, 0, :].detach().numpy().reshape(-1)
    for description in [d["description"] for d in data]
]

# Calculate cosine similarity between campaign purposes
cosine_sim = cosine_similarity(purpose_embeddings, purpose_embeddings)

# Define a function to get similar campaigns based on campaign purpose
def get_similar_campaigns(title, cosine_sim=cosine_sim, data=data):
    title_matches = [d for d in data if d["campaignName"] == title]
    if len(title_matches) > 0:
        idx = data.index(title_matches[0])
    else:
        raise HTTPException(status_code=404, detail=f"Campaign {title} not found")
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    campaign_indices = [i[0] for i in sim_scores]
    result = [{"id": data[i]["id"], "campaignName": data[i]["campaignName"]} for i in campaign_indices]
    return result

# Define the API endpoint
@app.get("/recommend")
async def recommend(title: str):
    try:
        similar_campaigns = get_similar_campaigns(title)
        return {"recommended_campaigns": similar_campaigns}
    except HTTPException as e:
        raise e

# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)