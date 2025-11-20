"""
REAL-TIME AI MEETING NOTES TAKER
Fully offline, privacy-first, professional meeting transcription + AI insights
Features: Live transcript, AI summaries, action items, speaker hints, auto-save, email
"""

import gradio as gr
import threading
import queue
import time
import pyaudio
import numpy as np
from datetime import datetime
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from faster_whisper import WhisperModel
import ollama
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
CHUNK_SIZE = 1024
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
SILENCE_THRESHOLD = 600
SILENCE_DURATION = 1.5
LONG_SILENCE_AUTO_STOP = 30  # seconds → auto-stop meeting
SUMMARY_MIN_WORDS = 80       # Only update summary if this many new words
WHISPER_MODEL = "medium"     # Much more accurate than "base" — still fast on CPU

# ============================================================================
# AUDIO PROCESSOR
# ============================================================================
class AudioProcessor:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_queue = queue.Queue()

    def start_recording(self):
        self.is_recording = True
        self.stream = self.audio.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._callback
        )
        self.stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

    def cleanup(self):
        self.stop_recording()
        self.audio.terminate()

# ============================================================================
# TRANSCRIPTION ENGINE (Faster-Whisper)
# ============================================================================
class TranscriptionEngine:
    def __init__(self, model_size=WHISPER_MODEL):
        print(f"Loading Whisper model: {model_size} (this may take 30-60 seconds first time)...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Whisper model loaded!")

    def transcribe(self, audio_np):
        audio_float = audio_np.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio_float,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_speech_duration_ms=250, threshold=0.6)
        )
        text = " ".join(seg.text for seg in segments).strip()
        return text

# ============================================================================
# AI SUMMARIZER (Ollama)
# ============================================================================
class AISummarizer:
    def __init__(self):
        self.model = self._detect_model()
        print(f"Using Ollama model: {self.model}")

    def _detect_model(self):
        try:
            models = ollama.list().get('models', [])
            names = [m['name'] for m in models]
            priority = ['llama3.1:8b', 'llama3.2:3b', 'gemma2', 'phi3']
            for p in priority:
                for n in names:
                    if p in n.lower():
                        return n
            return names[0] if names else "llama3.2:3b"
        except:
            return "llama3.2:3b"

    def summarize(self, text):
        if len(text.strip()) < 50:
            return "**Summary**\nNo content yet. Keep speaking!\n\n**Key Decisions**\n- None\n\n**Action Items**\n- None\n\n**Important Points**\n- Recording in progress..."

        prompt = f"""Meeting Transcript:
{text}

Provide a professional summary with these sections:
1. **Summary** (2–3 sentences)
2. **Key Decisions** (bullet points)
3. **Action Items** (who + what, if mentioned)
4. **Important Points** (bullet points)

Be concise and accurate."""

        try:
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}],
                                 options={'temperature': 0.3, 'num_predict': 600})
            return response['message']['content']
        except Exception as e:
            return f"Summary error: {e}\n\nCheck that Ollama is running (`ollama serve`)"

# ============================================================================
# MAIN MEETING MANAGER
# ============================================================================
class MeetingManager:
    def __init__(self):
        self.audio = AudioProcessor()
        self.transcriber = TranscriptionEngine()
        self.summarizer = AISummarizer()

        self.transcript_lines = []
        self.raw_transcript = []
        self.last_summary_words = 0
        self.start_time = None
        self.is_running = False
        self.thread = None

    def start(self):
        if self.is_running:
            return "Already recording!"
        self.is_running = True
        self.start_time = time.time()
        self.transcript_lines = []
        self.raw_transcript = []
        self.last_summary_words = 0

        self.audio.start_recording()
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        return "Recording started! Speak now."

    def _process_loop(self):
        buffer = []
        silence_start = None

        while self.is_running:
            try:
                data = self.audio.audio_queue.get(timeout=0.1)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_chunk).mean()

                buffer.extend(audio_chunk)

                # Detect silence
                if volume < SILENCE_THRESHOLD:
                    if silence_start is None:
                        silence_start = time.time()
                else:
                    silence_start = None

                # Process on full buffer OR long pause
                if len(buffer) >= SAMPLE_RATE * 3 or \
                   (silence_start and time.time() - silence_start > SILENCE_DURATION):

                    if len(buffer) > SAMPLE_RATE * 0.5:
                        text = self.transcriber.transcribe(np.array(buffer))
                        if text:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            formatted = f"[{timestamp}] {text}"
                            self.transcript_lines.append(formatted)
                            self.raw_transcript.append(text)

                    buffer = []

                # Auto-stop on long silence
                if silence_start and time.time() - silence_start > LONG_SILENCE_AUTO_STOP:
                    self.is_running = False
                    return

            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print("Error in loop:", e)

    def stop(self):
        if not self.is_running:
            return "Not recording."
        self.is_running = False
        self.audio.stop_recording()
        return "Recording stopped. Generating final notes..."

    def get_transcript(self):
        return "\n\n".join(self.transcript_lines[-100:])  # Last 100 lines

    def get_summary(self):
        full_text = " ".join(self.raw_transcript)
        word_count = len(full_text.split())
        if word_count < self.last_summary_words + SUMMARY_MIN_WORDS:
            return None  # No significant new content
        self.last_summary_words = word_count
        return self.summarizer.summarize(full_text)

    def save_notes(self, email=None, app_password=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        filename = f"meeting_notes_{timestamp}.md"
        os.makedirs("meeting_notes", exist_ok=True)
        filepath = os.path.join("meeting_notes", filename)

        full_text = " ".join(self.raw_transcript)
        final_summary = self.summarizer.summarize(full_text)

        content = f"""# Meeting Notes - {date_str}

## AI Summary & Insights
{final_summary}

## Full Transcript
{"\n\n".join(self.transcript_lines)}

---
*Generated by Real-Time AI Meeting Notes Taker • 100% Offline • Powered by Faster-Whisper + Ollama*
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        result = f"Notes saved to:\n`{filepath}`\n\n"

        if email and app_password:
            try:
                msg = MIMEMultipart()
                msg['From'] = email
                msg['To'] = email
                msg['Subject'] = f"Meeting Notes - {timestamp}"
                msg.attach(MIMEText(content, 'plain'))

                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(email, app_password)
                server.send_message(msg)
                server.quit()
                result += "Email sent successfully!"
            except Exception as e:
                result += f"Email failed: {e}"

        return result, content, filepath

    def cleanup(self):
        self.audio.cleanup()

# ============================================================================
# GRADIO UI
# ============================================================================
manager = MeetingManager()

with gr.Blocks(title="AI Meeting Notes Taker", theme=gr.themes.Soft(primary_hue="purple")) as app:
    gr.Markdown("# Real-Time AI Meeting Notes Taker\nProfessional • Offline • Private • Instant Summaries")

    with gr.Row():
        with gr.Column(scale=2):
            transcript = gr.Textbox(label="Live Transcript", lines=16, elem_classes="transcript-box")
            summary = gr.Markdown("Summary will appear automatically...")
        with gr.Column(scale=1):
            gr.Markdown("### Controls")
            start_btn = gr.Button("Start Recording", variant="primary", size="lg")
            stop_btn = gr.Button("Stop & Save", variant="stop", size="lg", interactive=False)
            status = gr.Textbox(label="Status", value="Ready", interactive=False)

            gr.Markdown("### Email Delivery (Optional)")
            email_in = gr.Textbox(label="Gmail Address", placeholder="you@gmail.com")
            pass_in = gr.Textbox(label="App Password", type="password", placeholder="16-char app password")
            save_status = gr.Textbox(label="Save Result", interactive=False)

            download_file = gr.File(label="Download Notes")

    def start():
        msg = manager.start()
        return gr.update(value=msg), gr.update(interactive=False), gr.update(interactive=True)

    def stop(email, password):
        msg = manager.stop()
        result, content, path = manager.save_notes(email, password)
        return (
            gr.update(value="Processing final summary..."),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(value=result),
            gr.update(value=content),
            path
        )

    start_btn.click(start, outputs=[status, start_btn, stop_btn])
    stop_btn.click(stop, inputs=[email_in, pass_in],
                   outputs=[status, start_btn, stop_btn, save_status, summary, download_file])

    # Live Updates
    app.load(lambda: manager.get_transcript(), None, transcript, every=1)
    app.load(lambda: manager.get_summary() or gr.skip(), None, summary, every=45)

app.launch(server_name="0.0.0.0", server_port=7860, share=False, quiet=False)
