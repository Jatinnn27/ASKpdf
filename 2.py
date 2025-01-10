from google.auth.transport.requests import Request
from google.oauth2 import service_account
import google.generativeai as genai

key_path = "**************************"
scopes = ["https://www.googleapis.com/auth/generative-language"]

credentials = service_account.Credentials.from_service_account_file(key_path, scopes=scopes)
credentials.refresh(Request())
genai.configure(api_key=None, credentials=credentials)

# Test API call
try:
    response = genai.generate_text(prompt="Hello world!", model="text-bison-001")
    print("API Call Success:", response)
except Exception as e:
    print("Error:", e)
