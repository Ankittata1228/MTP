
import os
import pandas as pd
import streamlit as st
from pytube import YouTube
import requests
from pydub import AudioSegment
from pydub.silence import split_on_silence  # Import split_on_silence from pydub.silence
from googletrans import Translator
import whisper
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")
from gtts import gTTS


# Define multiple API keys
api_keys = ['62175cad24ce47fd94b6c28ede5f07f8', 'e3e71f1a40d54573a22e1f7677c5a351']
current_api_key_index = 0

def get_next_api_key():
    global current_api_key_index
    current_api_key = api_keys[current_api_key_index]
    current_api_key_index = (current_api_key_index + 1) % len(api_keys)
    return current_api_key

def get_headers():
    api_key = get_next_api_key()
    headers = {"authorization": api_key, "content-type": "application/json"}
    return headers

if 'status' not in st.session_state:
    st.session_state['status'] = 'submitted'

transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
upload_endpoint = 'https://api.assemblyai.com/v2/upload'

# headers_auth_only = {'authorization': auth_key}
# headers = {
#     "authorization": auth_key,
#     "content-type": "application/json"
# }
CHUNK_SIZE = 2097152

# Initialize Translator
translator = Translator()

@st.cache_data
def download_video(link, output_directory="."):
    try:
        yt = YouTube(link)
        stream = yt.streams.filter(only_audio=True).first()
        file_name = f"{yt.video_id}.mp4"  # Download as MP4 to ensure best quality
        file_path = os.path.join(output_directory, file_name)
        stream.download(output_path=output_directory, filename=f"{yt.video_id}.mp4")
        return file_path
    except IndexError as e:
        st.error(f"Error downloading video: {e}")
        return None


@st.cache_data
def transcribe_and_save(link, audio_output_directory="./Audio_Files", transcription_output_directory="./Transcription", hindi_audio_output_directory="./Audio_Files/hindi_audio_file"):
    try:        
        video_file_path = download_video(link, audio_output_directory)
        if video_file_path is None:
            return
    except Exception as e:
        st.error(f"Error during download: {e}")
        return

    st.write('Saved mp4 to', video_file_path)

    # Convert video to audio (MP3)
    audio_file_path = convert_video_to_mp3(video_file_path, audio_output_directory)
    audio_segments = segment_audio_on_silence(audio_file_path)
    save_transcriptions_and_translations(audio_segments, audio_output_directory, transcription_output_directory, hindi_audio_output_directory)

def speech_to_text(audio_file_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file_path)
    return result["text"]

def translate_text(text, target_language='hi'):
    translation = translator.translate(text, dest=target_language)
    return translation.text
        # translation = translator.translate(text, dest=target_language)
        # print("Translation result:", translation)
        # translated_parts = list(map(lambda part: TranslatedPart(part[0], part[1] if len(part) >= 2 else []), parsed[1][0][0][5]))
        # return translated_parts

# def save_transcriptions_and_translations(audio_segments, audio_output_directory, transcription_output_directory, hindi_audio_output_directory):
#     data = {"Audio Segment": [], "Transcription": [], "Translation": [], "Hindi Audio Filename": []}
    
    
    
#     for i, segment in enumerate(audio_segments):

#         # Save audio segment
#         with open('number.txt', 'r') as file:
#     # Read the content of the file
#             index = file.read()
#             file.close()
#         st.write(f"Processing segment {index}...")
        

#         audio_segment_name = f"audio_segment_{index}.wav"
#         audio_segment_path = os.path.join(audio_output_directory, audio_segment_name)
#         segment.export(audio_segment_path, format="wav")
#         data["Audio Segment"].append(audio_segment_name)

#         # Transcribe
#         text = speech_to_text(audio_segment_path)
#         data["Transcription"].append(text if text else "Transcription not available")

#         # Translate
#         translated_text = translate_text(text, target_language='hi')
#         data["Translation"].append(translated_text)

       
#         if translated_text and translated_text != "Translation not available":
#             hindi_audio_name = f"hindi_audio_{index}.mp3"
#             hindi_audio_path = os.path.join(hindi_audio_output_directory, hindi_audio_name)

            
#             os.makedirs(hindi_audio_output_directory, exist_ok=True)

           
#             tts = gTTS(translated_text, lang='hi', slow=False)

#             try:
                
#                 tts.save(hindi_audio_path)
#                 data["Hindi Audio Filename"].append(hindi_audio_name)
#             except Exception as e:
#                 st.error(f"Error saving Hindi audio: {e}")
#         with open('number.txt', 'w') as file:
#             # Write the new value to the file
#             new_value = str(int(index)+1) # Replace this with the value you want to write
#             file.write(new_value)
#             file.close()

   
#     lengths = {key: len(value) for key, value in data.items()}
#     if len(set(lengths.values())) != 1:
#         st.error(f"Error: Inconsistent lengths of lists in data dictionary: {lengths}")
#         return

#     # Create DataFrame
#     df = pd.DataFrame(data)
#     csv_file_path = os.path.join(transcription_output_directory, 'transcriptions_and_translations.csv')
#     df.to_csv(csv_file_path, index=False)
#     st.write("Audio segments saved to:", audio_output_directory)
#     st.write("Transcriptions and translations saved to:", transcription_output_directory)
#     st.write("Hindi audio saved to:", hindi_audio_output_directory)
#     st.write("CSV file saved to:", csv_file_path)
def save_transcriptions_and_translations(audio_segments, audio_output_directory, transcription_output_directory, hindi_audio_output_directory):
    csv_file_path = os.path.join(transcription_output_directory, 'transcriptions_and_translations.csv')

    # Check if the CSV file already exists
    if os.path.exists(csv_file_path):
        # Read the existing CSV file
        existing_df = pd.read_csv(csv_file_path)
    else:
        # Create an empty DataFrame if the CSV file doesn't exist
        existing_df = pd.DataFrame(columns=["Audio Segment", "Transcription", "Translation", "Hindi Audio Filename"])

    for i, segment in enumerate(audio_segments):
        # Save audio segment
        with open('number.txt', 'r') as file:
            # Read the content of the file
            index = file.read()
            file.close()
        st.write(f"Processing segment {index}...")

        audio_segment_name = f"audio_segment_{index}.wav"
        audio_segment_path = os.path.join(audio_output_directory, audio_segment_name)
        segment.export(audio_segment_path, format="wav")

        # Transcribe
        text = speech_to_text(audio_segment_path)

        # Translate
        translated_text = translate_text(text, target_language='hi')

        if translated_text and translated_text != "Translation not available":
            hindi_audio_name = f"hindi_audio_{index}.mp3"
            hindi_audio_path = os.path.join(hindi_audio_output_directory, hindi_audio_name)

            os.makedirs(hindi_audio_output_directory, exist_ok=True)

            tts = gTTS(translated_text, lang='hi', slow=False)

            try:
                tts.save(hindi_audio_path)
            except Exception as e:
                st.error(f"Error saving Hindi audio: {e}")
                continue  # Skip to the next iteration if there's an error

            # Append the new data to the existing DataFrame
            new_data = {"Audio Segment": [audio_segment_name], "Transcription": [text if text else "Transcription not available"],
                        "Translation": [translated_text], "Hindi Audio Filename": [hindi_audio_name]}
            existing_df = pd.concat([existing_df, pd.DataFrame(new_data)], ignore_index=True)

        # Update the index in 'number.txt'
        with open('number.txt', 'w') as file:
            new_value = str(int(index) + 1)
            file.write(new_value)

    # Check for inconsistent lengths of lists in data dictionary
    lengths = {key: len(value) for key, value in existing_df.items()}
    if len(set(lengths.values())) != 1:
        st.error(f"Error: Inconsistent lengths of lists in data dictionary: {lengths}")
        return

    # Save the updated DataFrame to the CSV file
    existing_df.to_csv(csv_file_path, index=False)

    st.write("Audio segments saved to:", audio_output_directory)
    st.write("Transcriptions and translations saved to:", transcription_output_directory)
    st.write("Hindi audio saved to:", hindi_audio_output_directory)
    st.write("CSV file updated at:", csv_file_path)



def get_status(polling_endpoint):
    polling_response = requests.get(polling_endpoint, headers=get_headers())
    st.session_state['status'] = polling_response.json().get('status')

def refresh_state():
    st.session_state['status'] = 'submitted'

def read_file(filename):
    with open(filename, 'rb') as file:
        while True:
            data = file.read(CHUNK_SIZE)
            if not data:
                break
            yield data

def convert_video_to_mp3(video_file_path, output_directory):
    audio_file_name = os.path.splitext(os.path.basename(video_file_path))[0] + ".mp3"
    audio_file_path = os.path.join(output_directory, audio_file_name)

    # Convert video to audio
    audio = AudioSegment.from_file(video_file_path, format="mp4")
    audio.export(audio_file_path, format="mp3")

    return audio_file_path

def segment_audio_on_silence(audio_file_path, max_segment_duration=20 * 1000, min_silence_len=600, silence_thresh=-40):
    audio = AudioSegment.from_file(audio_file_path)

    # Split audio on silence
    segments = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # Filter segments based on max duration
    filtered_segments = [segment for segment in segments if len(segment) <= max_segment_duration]

    return filtered_segments
# def segment_audio_on_silence(audio_file_path, min_silence_len=600, silence_thresh=-40):
#     audio = AudioSegment.from_file(audio_file_path)

#     # Split audio on silence
#     segments = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

#     return segments

st.title("Hindi to English Dataset Creation Interface")
video_url = st.text_input("Enter YouTube Video URL:")


if st.button("Transcribe and Save"):
    transcribe_and_save(video_url, audio_output_directory="./Audio_Files", transcription_output_directory="./Transcription", hindi_audio_output_directory="./Audio_Files/hindi_audio_files")
