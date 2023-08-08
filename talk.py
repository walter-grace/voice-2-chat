import os
import tempfile
import base64
import streamlit as st
from elevenlabs import generate, set_api_key, save
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Set API keys
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")
set_api_key(os.getenv("ELEVENLABS_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")

# Title of the web app
st.title('Eleven Labs Demo')

def autoplay_audio(audio_data: bytes, format: str):
    b64 = base64.b64encode(audio_data).decode()
    st.markdown(
        f"""
        <audio controls autoplay>
        <source src="data:audio/{format};base64,{b64}" type="audio/{format}">
        </audio>
        """,
        unsafe_allow_html=True,
    )

# Input text
st.write("Type your message:")
user_input = st.text_input('')

if user_input:
    with st.spinner('Generating response...'):
        # Generate response from OpenAI GPT-3.5
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )

    with st.spinner('Generating audio...'):
        # Generate audio from the GPT-3.5 response using Elevenlabs' generate function
        audio_data = generate(
            text=completion.choices[0].message['content'],
            voice="Bella",
            model="eleven_monolingual_v1"
        )

        # Play the generated audio
        autoplay_audio(audio_data, 'wav')

        # Save the audio to a file
        filename = "audio_output.wav"
        save(audio_data, filename)

        # Display GPT-3.5's response and a download link
        st.write(f"{completion.choices[0].message['content']}")
        st.text('Refresh Page to start a new conversation')
