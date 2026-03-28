# 🏥 Rural Health Triage Assistant

> **AI-powered offline symptom checker for rural health workers and villagers.**
> Runs on any machine — no internet needed after setup.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🩺 **Symptom Triage** | XGBoost classifier diagnoses 10 common diseases with 95% accuracy |
| 🌐 **Bilingual** | Full English + Hindi support — UI labels, advice, and output all toggle dynamically |
| 🎤 **Voice Input** | Offline Hindi speech recognition via Vosk ASR |
| 🔊 **Text-to-Speech** | Advice read aloud via pyttsx3 (offline) |
| 🧠 **Explainability** | SHAP-powered "Most influential symptoms" shown with every result |
| 🔴 **Red-Flag Rules** | Unconsciousness, seizures, severe bleeding, chest pain+sweating → instant RED alert |
| 📞 **Ambulance Banner** | Prominent "Call 108" banner on emergency results |
| 🏥 **Nearest Clinics** | Offline clinic database with distance and phone numbers |
| 🔒 **Privacy** | 100% offline, no data stored, no cloud calls |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (~10 sec, creates models/ folder)
python train_model.py

# 3. Launch the UI at http://localhost:7860
python app.py
```

### Optional: REST API
```bash
pip install fastapi uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000
# POST /triage  •  GET /health
```

---

## 📁 Project Structure

```
AI_Health_adviser/
├── app.py                  # Gradio web UI (main app)
├── api.py                  # FastAPI REST endpoint
├── train_model.py          # Model training (synthetic data + XGBoost)
├── voice.py                # Vosk offline speech-to-text
├── clinics.json            # Clinic database (editable)
├── test_cases.json         # 12 structured test cases
├── requirements.txt        # Python dependencies
├── models/
│   ├── triage_model.json   # Trained XGBoost model (~1.8 MB)
│   ├── metadata.json       # Feature/disease/urgency mappings
│   └── evaluation_report.json  # Accuracy, F1, confusion matrix
└── vosk-model-hi-0.22/     # Hindi ASR model (~50 MB, download separately)
```

---

## 🧪 Diseases Covered

| Disease | Urgency | Key Symptoms |
|---|---|---|
| Malaria | 🔴 RED | Fever, chills, sweating, headache |
| Dengue | 🔴 RED | High fever, rash, joint/eye pain |
| Pneumonia | 🔴 RED | Cough, fever, breathing difficulty |
| Heat Stroke | 🔴 RED | High fever, confusion, dizziness |
| Typhoid | 🟡 YELLOW | Fever, abdominal pain, weakness |
| Tuberculosis | 🟡 YELLOW | Chronic cough, night sweats, weight loss |
| Gastroenteritis | 🟡 YELLOW | Diarrhea, vomiting, stomach pain |
| Anemia | 🟡 YELLOW | Fatigue, pale skin, dizziness |
| Common Cold | 🟢 GREEN | Runny nose, sneezing, sore throat |
| Influenza | 🟢 GREEN | Fever, body ache, cough |

---

## 📊 Model Performance

- **Accuracy:** 95.1% (on test set of 1,000 cases)
- **Training data:** 5,000 synthetic cases (500 per disease)
- **Model size:** ~1.8 MB (XGBoost JSON)
- **Inference time:** <0.5s on CPU

---

## 🔊 Voice Setup (Optional)

Download the Hindi Vosk model from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) and place it at:
```
vosk-model-hi-0.22/vosk-model-hi-0.22/
```

---

## 🎯 SDG Alignment

- **SDG 3** — Good Health & Well-being: early diagnosis, reduced preventable deaths
- **SDG 9** — Innovation: affordable tech bridging urban/rural health gaps
- **SDG 10** — Reduced Inequalities: accessible to low-literacy users

---

## ⚠️ Disclaimer

This is a **health information tool**, not a medical device. It provides guidance only — always consult a health professional if symptoms worsen.
