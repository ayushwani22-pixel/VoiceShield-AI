\# 🛡️ VoiceShield AI — Real-Time Deepfake Voice Detection



VoiceShield AI is an anti-spoofing gateway designed to safeguard voice-authorized transactions and call centers against neural voice cloning and synthetic audio fraud. Powered by a fine-tuned \*\*Wav2Vec2\*\* acoustic transformer and DSP pre-processing, the platform classifies speech authenticity in sub-second intervals and blocks unauthorized actions.



\---



\## ⚡ Key Features



\- \*\*Dual Ingestion Modes:\*\* Test audio either via live microphone recording (3–10s) or through audio file upload (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`).

\- \*\*Acoustic Pre-Processing Pipeline:\*\* Standardizes arbitrary incoming rates to 16 kHz mono, trims non-speech room silence, and normalizes dynamic range using Librosa.

\- \*\*Chunked Transformer Inference:\*\* Evaluates long-form speech in standardized 10-second segments to detect vocoder anomalies and spectral distortions.

\- \*\*Risk-Based UI Gateway:\*\* Renders interactive Plotly waveforms, real-time risk gauges, and automates downstream transaction gating (locks transfer actions when fake probability $\\ge 65\\%$).

\- \*\*Local/Cloud Hybrid Fallback:\*\* Automatically switches to the base Hugging Face model checkpoint (`garystafford/wav2vec2-deepfake-voice-detector`) if local weights are absent.



\---



\## 🛠️ Architecture \& Tech Stack



| Layer | Component / Technology | Purpose |

| :--- | :--- | :--- |

| \*\*Frontend UI\*\* | Streamlit, Plotly | Interactive dashboard, session persistence, metric cards |

| \*\*DSP Engine\*\* | Librosa, NumPy | 16 kHz resampling, silence trimming, peak normalization |

| \*\*Core Model\*\* | PyTorch, Hugging Face Transformers | Fine-tuned `Wav2Vec2` for Audio Classification |

| \*\*Inference Logic\*\* | Chunked Softmax Logits Aggregation | Multi-segment probability averaging |



\---



\## 🚀 Getting Started



\### 1. Prerequisites

Ensure you have Python 3.9+ installed on your machine.



\### 2. Clone the Repository

```bash

git clone \[https://github.com/](https://github.com/)ayushwani22-pixel/VoiceShield-AI.git

cd VoiceShield-AI









\## 📁 Repository Structure



```text

├── app1.py               # Streamlit application entry point \& inference logic

├── train\_model.py        # Model fine-tuning script

├── create\_labels.py      # Dataset label generator

├── requirements.txt      # Python dependencies

├── .gitignore            # Git exclusion rules (weights, virtual envs, raw datasets)

└── README.md             # Project documentation





⚠️ Disclaimer

This system is an anti-spoofing research prototype developed for academic and competition demonstration. Classification accuracy may vary across unseen voice-synthesis models, aggressive audio compression, extreme telephone line noise, or clips under 0.5 seconds.



