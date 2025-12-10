"""
Threshold Testing Script for Photo Quality Validation
Automatically tests different threshold combinations to find optimal values

Usage:
1. Organize your photos:
   test_data/
     good/  <- Put good quality photos here
     bad/   <- Put poor quality photos here
2. Run: python test_thresholds.py
3. Check results in threshold_results.csv
"""

import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd
from insightface.app import FaceAnalysis

# Initialize InsightFace
print("Loading InsightFace model...")
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("✓ Model loaded\n")

def compute_blur_score(image):
    """Compute Laplacian variance (blur metric)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compute_brightness(image):
    """Compute average brightness"""
    return np.mean(image)

def analyze_photo(image_path):
    """Analyze a single photo and return quality metrics"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    # Detect faces
    faces = app.get(img)
    
    if len(faces) == 0:
        return {
            'path': image_path.name,
            'face_count': 0,
            'face_size': 0,
            'detection_score': 0,
            'blur_score': 0,
            'brightness': 0
        }
    
    # Get best face
    face = max(faces, key=lambda f: f.det_score)
    bbox = face.bbox.astype(int)
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    
    # Extract face crop for blur/brightness analysis
    face_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    return {
        'path': image_path.name,
        'face_count': len(faces),
        'face_size': min(face_width, face_height),
        'detection_score': float(face.det_score),
        'blur_score': compute_blur_score(face_crop),
        'brightness': compute_brightness(face_crop)
    }

def test_threshold_combination(good_metrics, bad_metrics, thresholds):
    """
    Test a specific threshold combination
    Returns: (accuracy, true_positive, false_positive, true_negative, false_negative)
    """
    min_face_size, min_blur, min_brightness, max_brightness, min_det_score = thresholds
    
    tp = fp = tn = fn = 0
    
    # Test good photos (should pass)
    for metric in good_metrics:
        if metric is None:
            fn += 1  # Failed to process
            continue
            
        passed = (
            metric['face_count'] == 1 and
            metric['face_size'] >= min_face_size and
            metric['blur_score'] >= min_blur and
            metric['brightness'] >= min_brightness and
            metric['brightness'] <= max_brightness and
            metric['detection_score'] >= min_det_score
        )
        
        if passed:
            tp += 1  # Correctly accepted
        else:
            fn += 1  # Incorrectly rejected (bad!)
    
    # Test bad photos (should fail)
    for metric in bad_metrics:
        if metric is None:
            tn += 1  # Correctly rejected (no face)
            continue
            
        passed = (
            metric['face_count'] == 1 and
            metric['face_size'] >= min_face_size and
            metric['blur_score'] >= min_blur and
            metric['brightness'] >= min_brightness and
            metric['brightness'] <= max_brightness and
            metric['detection_score'] >= min_det_score
        )
        
        if passed:
            fp += 1  # Incorrectly accepted (bad!)
        else:
            tn += 1  # Correctly rejected
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    
    return accuracy, tp, fp, tn, fn

def main():
    # Paths
    good_dir = Path('test_data/good')
    bad_dir = Path('test_data/bad')
    
    if not good_dir.exists() or not bad_dir.exists():
        print("ERROR: Please create test_data/good and test_data/bad directories")
        print("Put high-quality photos in 'good' and low-quality photos in 'bad'")
        return
    
    # Collect images
    good_images = list(good_dir.glob('*.jpg')) + list(good_dir.glob('*.png'))
    bad_images = list(bad_dir.glob('*.jpg')) + list(bad_dir.glob('*.png'))
    
    print(f"Found {len(good_images)} good images")
    print(f"Found {len(bad_images)} bad images")
    print()
    
    # Analyze all photos (this takes time)
    print("Analyzing good quality photos...")
    good_metrics = [analyze_photo(img) for img in good_images]
    
    print("Analyzing bad quality photos...")
    bad_metrics = [analyze_photo(img) for img in bad_images]
    
    print("\n" + "="*60)
    print("METRIC STATISTICS")
    print("="*60)
    
    # Print statistics
    good_valid = [m for m in good_metrics if m and m['face_count'] > 0]
    if good_valid:
        print("\nGood photos metrics:")
        print(f"  Face size:       {np.mean([m['face_size'] for m in good_valid]):.1f} ± {np.std([m['face_size'] for m in good_valid]):.1f}")
        print(f"  Blur score:      {np.mean([m['blur_score'] for m in good_valid]):.1f} ± {np.std([m['blur_score'] for m in good_valid]):.1f}")
        print(f"  Brightness:      {np.mean([m['brightness'] for m in good_valid]):.1f} ± {np.std([m['brightness'] for m in good_valid]):.1f}")
        print(f"  Detection score: {np.mean([m['detection_score'] for m in good_valid]):.3f} ± {np.std([m['detection_score'] for m in good_valid]):.3f}")
    
    bad_valid = [m for m in bad_metrics if m and m['face_count'] > 0]
    if bad_valid:
        print("\nBad photos metrics:")
        print(f"  Face size:       {np.mean([m['face_size'] for m in bad_valid]):.1f} ± {np.std([m['face_size'] for m in bad_valid]):.1f}")
        print(f"  Blur score:      {np.mean([m['blur_score'] for m in bad_valid]):.1f} ± {np.std([m['blur_score'] for m in bad_valid]):.1f}")
        print(f"  Brightness:      {np.mean([m['brightness'] for m in bad_valid]):.1f} ± {np.std([m['brightness'] for m in bad_valid]):.1f}")
        print(f"  Detection score: {np.mean([m['detection_score'] for m in bad_valid]):.3f} ± {np.std([m['detection_score'] for m in bad_valid]):.3f}")
    
    print("\n" + "="*60)
    print("TESTING THRESHOLD COMBINATIONS")
    print("="*60 + "\n")
    
    # Define threshold combinations to test
    # Format: (min_face_size, min_blur, min_brightness, max_brightness, min_det_score)
    threshold_configs = [
        # Loose (accept more)
        (60, 40, 30, 230, 0.6),
        (70, 50, 40, 220, 0.7),
        # Medium
        (80, 60, 50, 210, 0.75),
        (90, 70, 50, 210, 0.8),
        (100, 80, 50, 210, 0.8),
        # Strict (reject more)
        (110, 90, 60, 200, 0.85),
        (120, 100, 60, 200, 0.9),
    ]
    
    results = []
    
    for idx, thresholds in enumerate(threshold_configs, 1):
        min_face_size, min_blur, min_brightness, max_brightness, min_det_score = thresholds
        
        accuracy, tp, fp, tn, fn = test_threshold_combination(good_metrics, bad_metrics, thresholds)
        
        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'config': idx,
            'min_face_size': min_face_size,
            'min_blur': min_blur,
            'min_brightness': min_brightness,
            'max_brightness': max_brightness,
            'min_det_score': min_det_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn
        })
        
        print(f"Config {idx}: Accuracy={accuracy:.1%}, F1={f1_score:.3f}, FP={fp}, FN={fn}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('threshold_results.csv', index=False)
    print(f"\n✓ Results saved to threshold_results.csv")
    
    # Find best configuration
    best_config = df.loc[df['f1_score'].idxmax()]
    print("\n" + "="*60)
    print("RECOMMENDED THRESHOLDS (Best F1 Score)")
    print("="*60)
    print(f"MIN_FACE_SIZE = {int(best_config['min_face_size'])}")
    print(f"MIN_BLUR = {int(best_config['min_blur'])}")
    print(f"MIN_BRIGHTNESS = {int(best_config['min_brightness'])}")
    print(f"MAX_BRIGHTNESS = {int(best_config['max_brightness'])}")
    print(f"MIN_DET_SCORE = {best_config['min_det_score']:.2f}")
    print(f"\nAccuracy: {best_config['accuracy']:.1%}")
    print(f"False Positives (bad photos accepted): {int(best_config['false_positive'])}")
    print(f"False Negatives (good photos rejected): {int(best_config['false_negative'])}")
    print("="*60)

if __name__ == '__main__':
    main()
