import os
import tempfile
import base64
import streamlit as st
from audio_recorder_streamlit import audio_recorder
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
st.title('Basic Assistant')

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

# Initialize session state if it doesn't exist
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'response' not in st.session_state:
    st.session_state.response = ""

# Record audio
st.write("Record a 15-second audio sample:")
audio_bytes = audio_recorder()

def generate_response():
    # Transcribe the recorded audio using Whisper API
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

        with st.spinner('Transcribing audio...'):
            with open(tmp_path, "rb") as audio_file:
                transcription_output = openai.Audio.transcribe("whisper-1", audio_file)

        # Display the transcription and store it in session state
        st.session_state.transcription = transcription_output['text']
        st.write(f"Transcription: {st.session_state.transcription}")

        with st.spinner('One Moment Thinking...'):
            # Generate response from OpenAI GPT-3.5
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": st.session_state.transcription}
                ]
            )
            st.session_state.response = completion.choices[0].message['content']

        # Generate audio from the GPT-3.5 response using Elevenlabs' generate function
        audio_data = generate(
            text=st.session_state.response,
            voice="Bella",
            model="eleven_monolingual_v1"
        )

        # Play the generated audio
        autoplay_audio(audio_data, 'wav')

        # Save the audio to a file
        filename = "audio_output.wav"
        save(audio_data, filename)

        # Display a message and a download link
        st.write(f"{st.session_state.response}")

        # Remove the temporary file
        os.remove(tmp_path)

if st.button("Submit", on_click=generate_response):
    st.text('Ask another question')
