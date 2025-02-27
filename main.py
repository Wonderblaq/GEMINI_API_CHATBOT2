from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI()

# Enable CORS to allow external access (e.g., from Flutter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from anywhere
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Gemini API Key from Environment Variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing! Set it in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")


# Define request model
class PromptRequest(BaseModel):
    prompt: str


@app.get("/")
def home():
    return {"message": "Welcome to the Breast Cancer Classifier AI Chatbot"}


@app.post("/ask_gemini/")
def ask_gemini(request: PromptRequest):
    """Send a structured prompt to Gemini AI and get a response"""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    structured_prompt = f"""
    You are a breast cancer expert AI assisting in diagnosis interpretation. 
    Your job is to explain the features of breast cancer classification, 
    provide insights into diagnosis results, and suggest follow-up actions. 

    **Guidelines:**
    - Only answer questions related to breast cancer.
    - Explain medical terms in simple words.
    - Provide follow-up recommendations but do NOT make direct diagnoses.
    - If the query is not about breast cancer, politely decline.
    - Answer questions related to symptoms, causes, and remedies for breast cancer.
    - Provide minimal details on breast cancer prevention.
    - Advise patients to seek immediate medical care if diagnosed as malignant.

    **User Query:** {request.prompt}
    """

    try:
        response = model.generate_content(structured_prompt)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run FastAPI with Uvicorn (for deployment)
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use Render's PORT or default to 8000
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=port)
