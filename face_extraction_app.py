# Face Extraction Service - Flask API with Standard InsightFace + Quality Validation
# EXACT same logic as Jetson for 100% embedding compatibility
# Install: pip install -r requirements.txt
# Run: python face_extraction_app.py

from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from insightface.app import FaceAnalysis

app = Flask(__name__)

# Models will be loaded lazily on first request to avoid startup memory spike
face_app = None

def get_face_app():
    """Lazy load InsightFace models on first request"""
    global face_app
    if face_app is None:
        print("Loading InsightFace models (first request)...")
        face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        print("✓ Models loaded successfully")
    return face_app

print("\n" + "="*50)
print("Face Extraction Service Ready!")
print("="*50)
print("Pipeline: Standard InsightFace (buffalo_s)")
print("Model Loading: LAZY (on first request)")
print("Quality Validation: ENABLED")
print("="*50 + "\n")

# QUALITY THRESHOLDS  (Adjusted for production use - more lenient)
MIN_FACE_SIZE = 120      # Minimum face size in pixels (reduced from 163)
MIN_BLUR = 50            # Minimum blur score (Laplacian variance) - adjusted for mobile cameras
MIN_BRIGHTNESS = 80      # Minimum brightness (0-255) (reduced from 105)
MAX_BRIGHTNESS = 180     # Maximum brightness (0-255) (increased from 126)
MIN_DET_SCORE = 0.65     # Minimum detection confidence (reduced from 0.78)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'face-extraction',
        'version': '4.0.0',
        'model': 'Standard InsightFace (buffalo_s)',
        'quality_validation': 'enabled',
        'thresholds': {
            'face_size': MIN_FACE_SIZE,
            'blur': MIN_BLUR,
            'brightness': f'{MIN_BRIGHTNESS}-{MAX_BRIGHTNESS}',
            'detection_score': MIN_DET_SCORE
        }
    })

@app.route('/extract', methods=['POST'])
def extract_embeddings():
    """
    Extract face embeddings from photos with quality validation
    """
    try:
        # Get photos from request
        data = request.json
        if not data or 'photos' not in data:
            return jsonify({'error': 'No photos provided'}), 400
        
        photos = data['photos']
        if not photos or len(photos) == 0:
            return jsonify({'error': 'Photos array is empty'}), 400
        
        embeddings = []
        confidences = []
        
        # Process each photo
        for idx, photo_b64 in enumerate(photos):
            try:
                # Strip data URI prefix if present
                if photo_b64.startswith('data:'):
                    photo_b64 = photo_b64.split(',', 1)[1]
                
                # Decode base64 image
                img_data = base64.b64decode(photo_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    return jsonify({
                        'error': f'Could not decode image {idx+1}',
                        'quality_issue': 'invalid_image',
                        'image_index': idx+1
                    }), 400
                
                # InsightFace Inference (Detection + Alignment + Extraction)
                app = get_face_app()
                faces = app.get(img)
                
                # VALIDATION 1: Face count
                if not faces or len(faces) == 0:
                    return jsonify({
                        'error': f'No face detected in image {idx+1}. Please ensure your face is clearly visible.',
                        'quality_issue': 'no_face',
                        'image_index': idx+1
                    }), 400
                
                if len(faces) > 1:
                    return jsonify({
                        'error': f'Multiple faces detected in image {idx+1}. Please ensure only one person is in the photo.',
                        'quality_issue': 'multiple_faces',
                        'face_count': len(faces),
                        'image_index': idx+1
                    }), 400
                
                # Get best face
                best_face = max(faces, key=lambda f: f.det_score)
                confidence = best_face.det_score
                bbox = best_face.bbox.astype(int)
                
                face_width = int(bbox[2] - bbox[0])
                face_height = int(bbox[3] - bbox[1])
                face_size = int(min(face_width, face_height))
                
                # VALIDATION 2: Face size
                if face_size < MIN_FACE_SIZE:
                    return jsonify({
                        'error': f'Face too small in image {idx+1} ({face_size}px). Please move closer to the camera.',
                        'quality_issue': 'face_too_small',
                        'face_size': face_size,
                        'minimum_required': MIN_FACE_SIZE,
                        'image_index': idx+1
                    }), 400
                
                # VALIDATION 3: Detection confidence
                if confidence < MIN_DET_SCORE:
                    return jsonify({
                        'error': f'Low detection confidence in image {idx+1} ({confidence:.2f}). Please use a clearer photo.',
                        'quality_issue': 'low_confidence',
                        'confidence': float(confidence),
                        'minimum_required': MIN_DET_SCORE,
                        'image_index': idx+1
                    }), 400
                
                # Extract face crop for blur/brightness analysis
                face_crop = img[max(0, bbox[1]):min(img.shape[0], bbox[3]), 
                               max(0, bbox[0]):min(img.shape[1], bbox[2])]
                
                if face_crop.size == 0:
                    return jsonify({
                        'error': f'Invalid face region in image {idx+1}',
                        'quality_issue': 'invalid_crop',
                        'image_index': idx+1
                    }), 400
                
                # VALIDATION 4: Blur detection
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                
                if blur_score < MIN_BLUR:
                    return jsonify({
                        'error': f'Image {idx+1} is too blurry ({blur_score:.1f}). Please hold still.',
                        'quality_issue': 'blurry',
                        'blur_score': float(blur_score),
                        'minimum_required': MIN_BLUR,
                        'image_index': idx+1
                    }), 400
                
                # VALIDATION 5: Brightness
                brightness = float(np.mean(face_crop))
                
                if brightness < MIN_BRIGHTNESS:
                    return jsonify({
                        'error': f'Image {idx+1} is too dark ({brightness:.1f}). Please use better lighting.',
                        'quality_issue': 'too_dark',
                        'brightness': float(brightness),
                        'acceptable_range': [MIN_BRIGHTNESS, MAX_BRIGHTNESS],
                        'image_index': idx+1
                    }), 400
                
                if brightness > MAX_BRIGHTNESS:
                    return jsonify({
                        'error': f'Image {idx+1} is overexposed ({brightness:.1f}). Please reduce lighting.',
                        'quality_issue': 'too_bright',
                        'brightness': float(brightness),
                        'acceptable_range': [MIN_BRIGHTNESS, MAX_BRIGHTNESS],
                        'image_index': idx+1
                    }), 400
                
                # All validations passed - extract embedding
                embedding = best_face.embedding
                
                # Normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                embeddings.append(embedding)
                confidences.append(float(confidence))
                
                print(f"✓ Image {idx+1}: PASS - size={face_size}px, blur={blur_score:.1f}, brightness={brightness:.1f}, conf={confidence:.2f}")
                
            except Exception as e:
                print(f"Error processing image {idx+1}: {str(e)}")
                return jsonify({'error': f'Processing failed for image {idx+1}: {str(e)}'}), 500
        
        # Check if we got any embeddings
        if len(embeddings) == 0:
            return jsonify({
                'error': 'No valid faces found in any photos'
            }), 400
        
        # Average embeddings from multiple photos
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Re-normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        avg_confidence = np.mean(confidences)
        
        return jsonify({
            'success': True,
            'embedding': avg_embedding.tolist(),
            'photosProcessed': len(embeddings),
            'averageQuality': float(avg_confidence)
        })
        
    except Exception as e:
        print(f"Error in extract_embeddings: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Face Extraction Service...")
    print("="*50)
    print(f"Endpoint: http://localhost:5001")
    print(f"Health: GET /health")
    print(f"Extract: POST /extract")
    print(f"Quality Validation: ENABLED")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
