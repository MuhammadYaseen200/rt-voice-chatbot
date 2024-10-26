# Import required libraries
import whisper
from gtts import gTTS
from groq import Groq
import gradio as gr
import soundfile as sf
from dotenv import load_dotenv
import os
from transformers import pipeline

# Load environment variables
load_dotenv()
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise ValueError("Please set your GROQ_API_KEY in a .env file or as an environment variable.")

GROQ_API_KEY = "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"

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Load Whisper model
try:
    whisper_model = whisper.load_model("base")
except Exception as e:
    print("Error loading Whisper model. Please check the installation and import settings.")
    raise e

# Initialize sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis")

# Initialize chat history and user preferences
chat_history = []
user_preferences = {
    "language": "en",  # Default language
    "voice": "en"      # Default voice
}

def transcribe_audio(audio):
    try:
        audio_path = "input.wav"
        sf.write(audio_path, audio[1], audio[0])  # Save audio buffer to file
        result = whisper_model.transcribe(audio_path)  # Transcribe the audio
        return result["text"]
    except Exception as e:
        return f"Error during transcription: {str(e)}"

def chat_with_llm(user_input):
    try:
        global chat_history
        chat_history.append({"role": "user", "content": user_input})  # Append user input to history
        
        # Call the LLM
        response = client.chat.completions.create(
            messages=chat_history,
            model="llama3-8b-8192"
        )
        
        assistant_response = response.choices[0].message.content
        chat_history.append({"role": "assistant", "content": assistant_response})  # Append assistant response to history
        
        # Detect sentiment and modify response based on sentiment
        sentiment = sentiment_analysis(user_input)[0]
        if sentiment['label'] == 'NEGATIVE':
            return f"I sense some negativity. How can I help you feel better? {assistant_response}"
        
        return assistant_response
    except Exception as e:
        return f"Error during chat with LLM: {str(e)}"

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang=user_preferences["language"])  # Convert text to speech
        tts.save("response.mp3")  # Save speech as an MP3 file
        return "response.mp3"  # Return the audio file path
    except Exception as e:
        return f"Error during text-to-speech conversion: {str(e)}"

def chatbot_pipeline(audio):
    user_text = transcribe_audio(audio)  # Transcribe the audio input
    print(f"User said: {user_text}")  # Debugging output
    
    if "Error" in user_text:
        return user_text  # Return the error message if transcription failed

    llm_response = chat_with_llm(user_text)  # Get response from LLM
    print(f"LLM response: {llm_response}")  # Debugging output
    
    response_audio = text_to_speech(llm_response)  # Convert LLM response to speech
    return response_audio  # Return audio file path for playback

# Create and launch the Gradio interface
interface = gr.Interface(
    fn=chatbot_pipeline,
    inputs=gr.Audio(type="numpy", label="Speak into the microphone"),  # Capture audio input
    outputs=gr.Audio(type="filepath", label="Chatbot Response"),  # Provide audio output as a file
    title="Advanced Real-time Voice-to-Voice Chatbot",
    description="Speak into the microphone and receive a spoken response. Ensure your microphone is working.",
)

# Launch the Gradio interface
interface.launch(share=True)
