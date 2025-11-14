import cv2
import numpy as np
import json
import logging
import torch
import threading  # <-- CRITICAL FIX: Import threading for concurrency management

# Language and Django Imports
from deep_translator import GoogleTranslator
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render

# Assuming these models exist in your .models file
from .models import DetectedObject, Score 

logger = logging.getLogger(__name__)

# --- CONFIGURATION & MODEL LOADING ---

LANG_CODE_MAP = {
    'tamil': 'ta',
    'telugu': 'te',
    'kannada': 'kn',
    'hindi': 'hi',
    'french': 'fr',
    'japanese': 'ja'
}

# Load YOLOv5 model once when Django starts
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4 # Confidence threshold

# Global variable to hold the latest detected words
detected_words = set() 
# --- CRITICAL FIX: Lock to protect detected_words during concurrent access ---
WORD_LOCK = threading.Lock() 
# --------------------------------------------------------------------------


# --- HELPER FUNCTIONS (Transliteration and Fallback) ---

def get_pronunciation(text, lang_code):
    """Converts non-English text to Romanized/English pronunciation using transliteration."""
    try:
        # Check if the language code is for Indic languages
        if lang_code in ['ta', 'te', 'kn', 'hi']:
            # The SCHEMES map uses the code (e.g., 'ta') to determine the script
            # We transliterate from the native script to ITRANS (Romanized phonetic)
            
            # Note: We need to map the lang_code to the correct Indic script name (e.g., 'ta' -> 'tamil') 
            # for the SCHEMES dictionary, if the keys are full names, 
            # but usually, the library handles this if the keys are the script codes.
            
            # For simplicity and robustness, let's map the code to the Script object:
            if lang_code == 'ta':
                script = sanscript.TAMIL
            elif lang_code == 'te':
                script = sanscript.TELUGU
            elif lang_code == 'kn':
                script = sanscript.KANNADA
            elif lang_code == 'hi':
                script = sanscript.DEVANAGARI
            else:
                return text # Should not happen

            # Transliterate the translated text from its script to ITRANS
            return transliterate(text, script, sanscript.ITRANS)

        elif lang_code == 'ja':
            # Requires pykakasi for Japanese: converts to Romaji
            import pykakasi
            kks = pykakasi.kakasi()
            # Convert text (which may be Kanji/Kana) to hepburn (Romaji)
            result = kks.convert(text)
            return ' '.join([item['hepburn'] for item in result])
        
        else:
            # For French, etc., use the translation itself as the pronunciation guide.
            return text 
    except Exception as e:
        logger.error(f"Transliteration error: {str(e)}")
        # Fallback to the original text on error
        return text

def get_simple_translations(word, lang_code):
    """Fallback simple translations for common objects (Placeholder)."""
    word = word.lower()
    translations = {
        'apple': {
            'ta': {'translated': 'ஆப்பிள்', 'pronunciation': 'aappil'},
        }
    }
    return translations.get(word, {}).get(lang_code, None)


# --- VIDEO STREAMING & DETECTION ---

def gen_frames():
    """Generates frames from the webcam with YOLOv5 detection overlays."""
    global detected_words
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img)

            detected_names = results.pandas().xyxy[0]['name'].tolist()

            # --- FIX: Use lock when updating the global set ---
            with WORD_LOCK:
                detected_words = set(detected_names) 
            # ---------------------------------------------------
            
            # --- DATABASE ACCESS COMMENTED OUT TO PREVENT THREAD CRASH ---
            # for name in detected_names:
            #     obj, created = DetectedObject.objects.get_or_create(name=name)
            #     if not obj.seen:
            #         obj.seen = True
            #         obj.save()
            # -------------------------------------------------------------

            results.render()
            annotated_frame = results.ims[0]
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def video_feed(request):
    """Django view to stream the video feed with detection."""
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


# --- API VIEWS ---

def get_detected_objects(request):
    """API view to return the latest detected objects from the global set."""
    global detected_words
    
    # --- FIX: Use lock when reading the global set ---
    with WORD_LOCK:
        objects_list = list(detected_words)
    # --------------------------------------------------
    
    return JsonResponse({'objects': objects_list}) 


@csrf_exempt
def translate_selected_object(request):
    """Handles the user's selected word for translation."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            word = data.get('word', '').strip()
            lang_code = data.get('language_code', '').lower().strip() 
            
            if not word or not lang_code:
                return JsonResponse({'error': 'Missing word or language code'}, status=400)

            translation = GoogleTranslator(source='auto', target=lang_code).translate(word)
            pronunciation = get_pronunciation(translation, lang_code)
            
            return JsonResponse({
                'translation': translation,
                'transliteration': pronunciation,
                'source_word': word
            })
        except Exception as e:
            logger.error(f"Error in translate_selected_object: {str(e)}")
            return JsonResponse({'error': f'Server processing error: {str(e)}'}, status=500)
    return JsonResponse({'error': 'Invalid method'}, status=405)


@csrf_exempt
def update_score_and_learn(request):
    """Updates the score and marks a word as learned (Gamification/Tracking)."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            word = data.get('word')
            lang = data.get('lang')
            
            # --- Score and Learn Logic (Requires database access) ---
            obj = DetectedObject.objects.filter(name__iexact=word).first()
            if obj:
                obj.learnt = True
                if lang not in obj.language_learntin:
                    obj.language_learntin.append(lang)
                obj.save()

                score, _ = Score.objects.get_or_create(id=1)
                score.points += 10
                score.save()

                return JsonResponse({'status': 'success', 'points': score.points})
            # -------------------------------------------------------
            return JsonResponse({'status': 'failed', 'error': 'Word not found'}, status=404)
        except Exception as e:
            logger.error(f"Error in update_score_and_learn: {str(e)}")
            return JsonResponse({'status': 'failed', 'error': str(e)}, status=500)
    return JsonResponse({'status': 'failed', 'error': 'Invalid method'}, status=405)


def translate_word(request):
    """Legacy API: Translates a word based on query parameters (GET)."""
    word = request.GET.get('word', '').strip()
    lang = request.GET.get('lang', '').lower().strip()

    lang_code = LANG_CODE_MAP.get(lang)

    try:
        translation = GoogleTranslator(source='auto', target=lang_code).translate(word)
        pronunciation = get_pronunciation(translation, lang_code)
        
        return JsonResponse({
            'translated': translation,
            'pronunciation': pronunciation
        })
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        simple_data = get_simple_translations(word, lang_code)
        if simple_data:
            return JsonResponse(simple_data)
        return JsonResponse({'translated': 'Translation service unavailable', 'pronunciation': ''}, status=503)


# --- PAGE RENDERING VIEWS ---

def index(request):
    """Renders the main index page."""
    return render(request, 'index.html')

def detection(request):
    """Renders the AR detection page (det.html)."""
    return render(request, 'det.html')

def word_list(request):
    """Renders the word list page."""
    objects = DetectedObject.objects.filter(seen=True)
    return render(request, 'list.html', {"detected_objects": objects})
def detection(request):
    # This view function should render your main detection screen (det.html)
    return render(request, 'det.html') 

LANG_NAME_MAP = {
    'ta': 'Tamil', 
    'te': 'Telugu', 
    'kn': 'Kannada', 
    'hi': 'Hindi', 
    'fr': 'French', 
    'ja': 'Japanese'
}
# ⭐ NEW HELPER FUNCTION (YOUR TRANSLATION LOGIC GOES HERE)
def get_translation_data(word, target_lang_code):
    """Fetches translation and pronunciation using the two-letter code."""
    translated_text = GoogleTranslator(source='auto', target=target_lang_code).translate(word)
    transliterated_text = get_pronunciation(translated_text, target_lang_code)

    return {
        'original_word': word.capitalize(),
        'translation': translated_text,
        'transliteration': transliterated_text,  # ✅ changed key name here
        'target_language_name': LANG_NAME_MAP.get(target_lang_code, 'English'),
        'target_language_code': target_lang_code,
    }


# ⭐ UPDATED VIEW FUNCTION
# ⭐ FINAL CORRECTED VIEW FUNCTION
def translate_selected_object(request, object_name):
    """
    Handles the request for a selected object and renders the dedicated detail page.
    """
    
    # ⭐ CRITICAL FIX: Use the correct session key 'target_language' 
    # and default to the two-letter code 'ta'.
    target_lang_code = request.session.get('target_language', 'ta')
    
    try:
        # 2. Fetch all data using the correct two-letter code
        context = get_translation_data(object_name, target_lang_code)
        
        # 3. RENDER THE NEW DEDICATED TEMPLATE
        # Note: The context dictionary already contains 'original_word', 'translation', 
        # 'pronunciation', and language names/codes needed by 'translation_detail.html'.
        return render(request, 'translation_detail.html', context)
        
    except Exception as e:
        logger.error(f"Translation/Data Fetch Error: {e}")
        # Provide error context for the template
        error_context = {
            'original_word': object_name.capitalize(),
            'target_language_name': LANG_NAME_MAP.get(target_lang_code, 'English'),
            'translation': 'Error: Could not retrieve translation.',
            'pronunciation': 'N/A'
        }
        return render(request, 'translation_detail.html', error_context)
@csrf_exempt # You might need this if you are not passing CSRF tokens via AJAX headers
def set_target_language(request):
    """Sets the target language preference in the user's session."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            lang_code = data.get('language_code')
            
            if lang_code:
                request.session['target_language'] = lang_code
                return JsonResponse({'status': 'success', 'language': lang_code})
            else:
                return JsonResponse({'status': 'error', 'message': 'Missing language code'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=405)

def handwriting_test(request, object_name):
    """
    Renders a page where the user writes the translated word by hand.
    """
    target_lang_code = request.session.get('target_language', 'ta')

    try:
        # Get the correct translation
        data = get_translation_data(object_name, target_lang_code)
        context = {
            'original_word': object_name,
            'correct_translation': data['translation'],
            'target_language': LANG_NAME_MAP.get(target_lang_code, 'Unknown'),
        }
        return render(request, 'handwriting_test.html', context)
    except Exception as e:
        logger.error(f"Error loading handwriting test: {e}")
        return render(request, 'handwriting_test.html', {
            'error': 'Could not load test. Please try again later.'
        })


@csrf_exempt
def check_handwriting(request):
    """
    Compares the user's handwritten input (recognized via MediaPipe/OpenCV) 
    with the correct translated word and returns an accuracy score.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_text = data.get('user_text', '').strip().lower()
            correct_text = data.get('correct_text', '').strip().lower()

            # Basic string similarity check using Levenshtein distance
            from difflib import SequenceMatcher
            accuracy = round(SequenceMatcher(None, user_text, correct_text).ratio() * 100, 2)

            return JsonResponse({'accuracy': accuracy})
        except Exception as e:
            logger.error(f"Error checking handwriting: {str(e)}")
            return JsonResponse({'error': 'Failed to check handwriting'}, status=500)
    return JsonResponse({'error': 'Invalid method'}, status=405)

import cv2
import numpy as np
import mediapipe as mp
from django.views.decorators.csrf import csrf_exempt

# --- Gesture Writing Feature ---

def gesture_write(request, object_name):
    """
    Renders the gesture-writing page where the user writes
    the translated word using hand gestures.
    """
    target_lang_code = request.session.get('target_language', 'ta')

    try:
        data = get_translation_data(object_name, target_lang_code)
        context = {
            'original_word': object_name,
            'translated_word': data['translation'],  # ✅ Pass translated word
            'target_language': LANG_NAME_MAP.get(target_lang_code, 'Unknown'),
        }
        return render(request, 'gesture_write.html', context)
    except Exception as e:
        logger.error(f"Error loading gesture write: {e}")
        return render(request, 'gesture_write.html', {
            'error': 'Could not load gesture writing page. Please try again later.'
        })
import pytesseract

# Replace this with the actual installation path of tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# views.py
import base64
from io import BytesIO
from PIL import Image
import pytesseract
from django.http import JsonResponse
from difflib import SequenceMatcher
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def check_gesture(request):
    """
    Process the gesture canvas, recognize French text using Tesseract,
    and calculate accuracy automatically against the correct translation.
    """
    if request.method == 'POST':
        try:
            # Parse incoming JSON
            data = json.loads(request.body)
            correct_word = data.get('correct_word', '').strip()
            image_data = data.get('image', '')

            if not image_data:
                return JsonResponse({'status': 'error', 'message': 'No image received'})

            # Convert base64 string to PIL Image
            header, encoded = image_data.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            image = Image.open(BytesIO(img_bytes))

            # Convert to grayscale
            image = image.convert("L")

            # Resize for better OCR recognition
            image = image.resize((256, 256), Image.LANCZOS)

            # Apply threshold to clean up image
            image = image.point(lambda x: 0 if x < 128 else 255, '1')

            # OCR: Recognize French text using Latin script
            predicted_text = pytesseract.image_to_string(
                image,
                lang='eng',       # <-- Changed to English/Latin script for French
                config='--psm 7'  # Treat the image as a single line of text
            ).strip()

            # Calculate similarity accuracy
            from difflib import SequenceMatcher
            accuracy = round(SequenceMatcher(None, predicted_text.lower(), correct_word.lower()).ratio() * 100, 2)

            return JsonResponse({
                'status': 'success',
                'predicted_word': predicted_text,
                'correct_word': correct_word,
                'accuracy': accuracy
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)
