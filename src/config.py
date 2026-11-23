import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

DEFAULT_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-flash-latest",
    "ollama/qwen2.5:3b"
]

def get_llm(model_name: str = DEFAULT_MODEL_NAME, temperature: float = DEFAULT_TEMPERATURE):
    """
    Initializes and returns the Chat model instance (Google Gemini API or Local Ollama/Qwen).
    """
    # Check if user requested an offline local model (e.g. qwen2.5:3b, ollama/qwen:3b, local/llama3)
    is_offline = (
        model_name.startswith("ollama/") or 
        model_name.startswith("local/") or 
        "qwen" in model_name.lower() or 
        "llama" in model_name.lower() or
        "mistral" in model_name.lower()
    )

    if is_offline:
        clean_model = model_name.replace("ollama/", "").replace("local/", "")
        if clean_model.lower() == "qwen3b" or clean_model.lower() == "qwen":
            clean_model = "qwen2.5:3b"

        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=clean_model, temperature=temperature, base_url=ollama_url)
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(model=clean_model, temperature=temperature, base_url=ollama_url)
            except ImportError:
                raise ImportError(
                    "langchain-ollama is required for offline execution.\n"
                    "Install it via: pip install langchain-ollama"
                )

    # Online Gemini API
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GOOGLE_API_KEY is missing or set to the placeholder in .env file.\n"
            "Please obtain a free API key from https://aistudio.google.com/ or select an offline model (e.g. 'ollama/qwen2.5:3b')."
        )
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key
    )
