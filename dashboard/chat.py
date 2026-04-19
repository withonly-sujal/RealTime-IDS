import os
import json
from google import genai
from google.genai import types

class ChatAgent:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.sessions = {}
        
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client()
        else:
            self.client = None

        # System prompt setting the AI's persona
        self.system_instruction = (
            "You are an expert Network Intrusion Detection System (IDS) security analyst. "
            "You assist the user by answering general questions about cybersecurity, networking, and intrusion detection. "
            "Keep responses concise and use Markdown formatting for readability."
        )

    def is_configured(self):
        return self.client is not None

    def query(self, message: str, session_id: str) -> str:
        if not self.is_configured():
            return "Gemini API key not found. Please set GEMINI_API_KEY in your .env file and restart the server."
            
        history = self.sessions.get(session_id, [])
        
        history.append({
            "role": "user",
            "parts": [{"text": message}]
        })
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.3
                )
            )
            
            ai_text = response.text
            
            history.append({
                "role": "model",
                "parts": [{"text": ai_text}]
            })
            
            if len(history) > 20:
                history = history[-20:]
                
            self.sessions[session_id] = history
            return ai_text
            
        except Exception as e:
            return f"❌ Error contacting Gemini API: {str(e)}"
