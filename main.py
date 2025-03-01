from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
from fastapi.middleware.cors import CORSMiddleware




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
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Define request model
class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "Welcome to the Breast Cancer Classifier AI Chatbot"}

@app.post("/ask_gemini/")
def ask_gemini(request: PromptRequest):
    """Send a structured prompt to Gemini AI and get a response"""
    try:
        structured_prompt = f"""
        You are a breast cancer expert AI assisting in diagnosis interpretation. 
        Your job is to explain the features of breast cancer classification, 
        provide insights into diagnosis results, and suggest follow-up actions. 

        **Guidelines:**
        - Only answer questions related to breast cancer.
        - Explain medical terms in simple words.
        - Provide follow-up recommendations but do NOT make direct diagnoses.
        - If the query is not about breast cancer, politely decline.
        - Answer questions related to symptoms, causes and remedies for breast cancer
        - Answer lightly or give out minimal details on prevention of breast cancer
        - Advice Patient to seek out immediate medical care once they are diagnosed as malignant

        **User Query:** {request.prompt}
        """
        response = model.generate_content(structured_prompt)
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}
