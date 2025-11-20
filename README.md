### Real-Time AI Meeting Notes Taker  
**100% Offline • Privacy-First • Instant Summaries & Action Items**

A beautiful, professional, real-time meeting transcription app powered by **Faster-Whisper** and **Ollama**.  
No data leaves your machine. Perfect for sensitive meetings, lawyers, doctors, executives, and anyone who values privacy.

![](https://img.shields.io/badge/status-production%20ready-success) ![](https://img.shields.io/badge/offline-100%25-blue) ![](https://img.shields.io/badge/models-local-brightgreen)

### Live Demo (Screenshot)
![App Screenshot](screenshot.png)  
*(Add a real screenshot later — highly recommended!)*

### Features
- Real-time microphone transcription (<1s latency)  
- Live word-by-word streaming display  
- AI-powered summaries updated automatically  
- Extracts **Key Decisions**, **Action Items**, and **Important Points**  
- Speaker timestamps (`[14:23:11]`)  
- Auto-stop after long silence (30s)  
- Saves notes as clean, timestamped Markdown files  
- Optional email delivery (Gmail App Password)  
- Beautiful dark UI with auto-scroll  
- Download notes directly from browser  
- Fully offline — nothing sent to the cloud  

### Tech Stack (All Local)
- **Transcription**: `faster-whisper` (medium model)  
- **AI Brain**: `ollama` + `llama3.1:8b` or `llama3.2:3b`  
- **UI**: Gradio (runs in browser)  
- **Audio**: PyAudio + FFmpeg  

### Setup (One-Time)

#### 1. Install System Dependencies
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
Download from: https://ffmpeg.org/download.html
2. Install & Run Ollama
Go to: https://ollama.com/download
After install, run in terminal:
Bashollama pull llama3.2:3b
# Or for better quality:
ollama pull llama3.1:8b
3. Install Python Packages
Bashpip install -r requirements.txt
4. Run the App
Bashpython meeting_notes_taker.py
Opens automatically at: http://localhost:7860
Usage

(Optional) Enter your Gmail + App Password to auto-email notes
Click "Start Recording"
Speak normally — watch live transcript + AI summary appear
When done, click "Stop & Save"
Notes saved to meeting_notes/ folder + optional email + download

Gmail App Password (Required for Email)

Enable 2-Factor Authentication on your Google account
Go to: https://myaccount.google.com/apppasswords
Generate a 16-character App Password
Paste it (not your real password) into the app

Output Example (meeting_notes_20250405_142301.md)
Markdown# Meeting Notes - April 05, 2025 at 02:23 PM

## AI Summary & Insights
**Summary**  
The team discussed Q2 roadmap and decided to prioritize the new analytics dashboard...

**Key Decisions**
- Launch MVP by May 15th
- Assign Sarah as lead designer

**Action Items**
- John: Prepare demo by Friday
- Sarah: Finalize UI mockups (due Apr 10)

**Important Points**
- Budget increased by 20%
- Client loved the prototype
Folder Structure
text├── meeting_notes_taker.py
├── requirements.txt
├── README.md
├── screenshot.png (optional)
└── meeting_notes/
    └── meeting_notes_20250405_142301.md
Recommended Ollama Models


ModelQualitySpeedRecommendationllama3.1:8bBestMediumHighly recommendedllama3.2:3bGreatFastDefault fallbackgemma2:9bGreatMediumAlternativephi3:mediumGoodVery FastLightweight
Troubleshooting

"No audio" → Check microphone permissions
"Model not found" → Run ollama list and pull a model
"Ollama error" → Start Ollama with ollama serve
Slow transcription → Use small or base Whisper model (edit code)

License
MIT License — feel free to use, modify, and share.
Made with ❤️ by Privacy-First AI Enthusiasts
Star this repo if you love offline AI!
Want a macOS/Windows app version? Let me know!

No cloud. No tracking. Just your voice → perfect notes.
