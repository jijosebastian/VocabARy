# VocabARy

This project is an AR/VR-based language learning system that detects real-world objects, translates their names into a selected language, and lets users practice writing translations using hand gestures. The system uses YOLOv5 for object detection, MediaPipe for gesture tracking, and Tesseract OCR for recognizing handwritten input. It now includes French language support and a modern, user-friendly interface for a more interactive learning experience.

---

## Features

- Real-time object detection using the camera.  
- Multilingual vocabulary overlay.  
- Pronunciation and transliteration for selected languages.  
- Browser-based interface, no installation required (except Python backend).  
- Supports multiple Indian and foreign languages.  

---

## Installation and Running steps :

```bash
git clone https://github.com/jijosebastian/VocabARy.git
cd VocabARy
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py runserver
