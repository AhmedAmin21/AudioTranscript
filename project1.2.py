import streamlit as st
import yt_dlp
import os
from groq import Groq
from dotenv import load_dotenv
from pydub import AudioSegment
import tempfile
import time
import gc  # Garbage collector to free up resources

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Set page title
st.title('Audio & YouTube Video Transcription')

# --- SESSION STATE INITIALIZATION ---
if 'yt_transcribe' not in st.session_state:
    st.session_state.yt_transcribe = False
    
if 'aud_transcribe' not in st.session_state:
    st.session_state.aud_transcribe = False
    
if 'yt_result' not in st.session_state:
    st.session_state.yt_result = None
    
if 'audio_result' not in st.session_state:
    st.session_state.audio_result = None
    
if 'processing' not in st.session_state:
    st.session_state.processing = False
    
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# --- FILE HANDLING FUNCTIONS (WINDOWS COMPATIBLE) ---
def safe_delete(file_path):
    """Safely delete files on Windows with retries and resource cleanup"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            return True
    except PermissionError:
        # Force garbage collection to release file handles
        gc.collect()
        time.sleep(1)
        try:
            os.unlink(file_path)
            return True
        except:
            st.warning(f"Couldn't delete temporary file: {file_path}")
            return False
    except Exception as e:
        st.warning(f"Error deleting file: {str(e)}")
        return False
    return False

# --- AUDIO PROCESSING FUNCTIONS ---
def split_audio(file_path, chunk_minutes=2):
    """Split audio into manageable chunks"""
    try:
        audio = AudioSegment.from_file(file_path)
        chunk_ms = chunk_minutes * 60 * 1000
        return [audio[i:i+chunk_ms] for i in range(0, len(audio), chunk_ms)]
    except Exception as e:
        st.error(f"Error splitting audio: {str(e)}")
        return []

def transcribe_chunk(chunk_file, retries=3):
    """Transcribe a single audio chunk with error handling"""
    for attempt in range(retries):
        try:
            with open(chunk_file, 'rb') as f:
                return client.audio.transcriptions.create(
                    file=(chunk_file, f.read()),
                    model="whisper-large-v3-turbo",
                    response_format="text"
                )
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 5 * (2 ** attempt)
                st.warning(f"Retrying chunk in {wait_time}s...")
                time.sleep(wait_time)
            else:
                st.error(f"Chunk failed: {str(e)}")
                return ""
    return ""

def process_long_audio(file_path):
    """Process long audio through chunked transcription"""
    try:
        chunks = split_audio(file_path)
        if not chunks:
            return ""
            
        full_transcript = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, chunk in enumerate(chunks):
            # Update progress
            progress = (i + 1) / len(chunks)
            st.session_state.progress = progress
            progress_bar.progress(progress)
            status_text.text(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Create temporary file safely
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = temp_file.name
                chunk.export(temp_path, format="mp3", bitrate="64k")
                
            # Transcribe and immediately clean up
            transcript = transcribe_chunk(temp_path)
            full_transcript.append(transcript)
            safe_delete(temp_path)  # Clean up immediately after use
            
        return "\n\n".join(full_transcript)
        
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return ""

# --- USER INTERFACE ---
col1, col2 = st.columns(2)

# --- AUDIO TRANSCRIPTION COLUMN ---
with col1:
    st.subheader("Audio File Transcription")
    
    if st.button("Upload Audio File"):
        st.session_state.yt_transcribe = False
        st.session_state.aud_transcribe = True
        st.session_state.audio_result = None
    
    if st.session_state.aud_transcribe:
        audio_file = st.file_uploader('Select audio file', 
                                     type=['mp3', 'wav', 'm4a'],
                                     label_visibility="collapsed")
        
        if audio_file and not st.session_state.processing:
            # Create safe temporary file path
            temp_path = None
            try:
                st.session_state.processing = True
                
                # Write to unique temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(audio_file.read())
                    temp_path = tmp_file.name
                
                # Process audio
                st.session_state.audio_result = process_long_audio(temp_path)
                
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
            finally:
                # Clean up temp file
                if temp_path:
                    safe_delete(temp_path)
                st.session_state.processing = False
            
            if st.session_state.audio_result:
                st.success("Audio processing complete!")
        
        if st.session_state.audio_result:
            st.download_button(
                "Download Transcript",
                st.session_state.audio_result,
                file_name="audio_transcript.txt"
            )
            st.text_area("Transcription Result", 
                         st.session_state.audio_result, 
                         height=300)

# --- YOUTUBE TRANSCRIPTION COLUMN ---
with col2:
    st.subheader("YouTube Video Transcription")
    
    if st.button("Enter YouTube URL"):
        st.session_state.aud_transcribe = False
        st.session_state.yt_transcribe = True
        st.session_state.yt_result = None
    
    if st.session_state.yt_transcribe:
        url = st.text_input("Paste YouTube URL:", 
                            placeholder="https://www.youtube.com/watch?v=...",
                            label_visibility="collapsed")
        
        if url:
            st.video(url)
            
            if st.button("Transcribe Video") and not st.session_state.processing:
                temp_path = None
                try:
                    st.session_state.processing = True
                    
                    # Download audio
                    with st.spinner("Downloading audio..."):
                        # Use unique filename pattern
                        ydl_opts = {
                            'format': 'bestaudio/best',
                            'outtmpl': 'yt_audio_%(id)s.%(ext)s',  # Unique filename
                            'postprocessors': [{
                                'key': 'FFmpegExtractAudio',
                                'preferredcodec': 'mp3',
                                'preferredquality': '64',
                            }],
                        }
                        
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(url, download=False)
                            safe_filename = ydl.prepare_filename(info)
                            audio_filename = safe_filename.rsplit('.', 1)[0] + '.mp3'
                            
                            # Only download if needed
                            if not os.path.exists(audio_filename):
                                ydl.download([url])
                            
                        temp_path = audio_filename
                    
                    # Process audio
                    st.session_state.yt_result = process_long_audio(temp_path)
                    
                except Exception as e:
                    st.error(f"YouTube processing error: {str(e)}")
                finally:
                    # Clean up downloaded file
                    if temp_path and os.path.exists(temp_path):
                        safe_delete(temp_path)
                    st.session_state.processing = False
                
                if st.session_state.yt_result:
                    st.success("YouTube processing complete!")
        
        if st.session_state.yt_result:
            st.download_button(
                "Download Transcript",
                st.session_state.yt_result,
                file_name="youtube_transcript.txt"
            )
            st.text_area("YouTube Transcription", 
                         st.session_state.yt_result, 
                         height=300)

# --- FOOTER ---
st.markdown("---")
st.caption("Note: Processing time depends on video/audio length")
st.caption("Temporary files are automatically cleaned up after processing")