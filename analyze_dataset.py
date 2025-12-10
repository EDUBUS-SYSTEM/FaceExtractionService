"""
Automated Dataset Analysis for Threshold Tuning
Analyzes the existing face-id-project dataset to find optimal quality thresholds

Dataset structure:
  images_dataset/
    ST001/  (100 photos)
    ST002/  (100 photos)
    ...
    ST059/  (33 photos)

This script will:
1. Analyze ALL photos in the dataset
2. Compute quality metrics (blur, brightness, face size, etc.)
3. Auto-categorize into good/bad based on percentiles
4. Test threshold combinations
5. Output recommended thresholds

Expected runtime: 10-15 minutes for ~3,700 images
"""

import cv2
import numpy as np
import os
from pathlib import Path
import pandas as pd
from insightface.app import FaceAnalysis
from tqdm import tqdm
import json

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
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Detect faces
        faces = app.get(img)
        
        if len(faces) == 0:
            return {
                'path': str(image_path),
                'person_id': image_path.parent.name,
                'face_count': 0,
                'face_size': 0,
                'detection_score': 0,
                'blur_score': 0,
                'brightness': 0,
                'quality_category': 'no_face'
            }
        
        # Get best face
        face = max(faces, key=lambda f: f.det_score)
        bbox = face.bbox.astype(int)
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        # Extract face crop
        face_crop = img[max(0, bbox[1]):min(img.shape[0], bbox[3]), 
                        max(0, bbox[0]):min(img.shape[1], bbox[2])]
        
        if face_crop.size == 0:
            return None
        
        return {
            'path': str(image_path),
            'person_id': image_path.parent.name,
            'face_count': len(faces),
            'face_size': min(face_width, face_height),
            'detection_score': float(face.det_score),
            'blur_score': compute_blur_score(face_crop),
            'brightness': compute_brightness(face_crop),
            'quality_category': 'good'  # Will be updated later
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def categorize_quality(df):
    """
    Auto-categorize photos into good/bad based on metric percentiles
    Bottom 30% of each metric = bad
    """
    # Filter only photos with detected faces
    df_faces = df[df['face_count'] > 0].copy()
    
    # Calculate percentiles
    p30_blur = df_faces['blur_score'].quantile(0.30)
    p30_brightness_low = df_faces['brightness'].quantile(0.15)
    p30_brightness_high = df_faces['brightness'].quantile(0.85)
    p30_face_size = df_faces['face_size'].quantile(0.30)
    p30_det_score = df_faces['detection_score'].quantile(0.30)
    
    print(f"\n30th Percentile Thresholds:")
    print(f"  Blur score:      {p30_blur:.1f}")
    print(f"  Face size:       {p30_face_size:.1f}px")
    print(f"  Brightness:      {p30_brightness_low:.1f} - {p30_brightness_high:.1f}")
    print(f"  Detection score: {p30_det_score:.3f}")
    
    # Mark as "bad" if ANY metric is in bottom 30%
    df.loc[:, 'quality_category'] = 'good'
    df.loc[df['face_count'] == 0, 'quality_category'] = 'no_face'
    df.loc[df['face_count'] > 1, 'quality_category'] = 'multiple_faces'
    
    bad_mask = (
        (df['blur_score'] < p30_blur) |
        (df['face_size'] < p30_face_size) |
        (df['brightness'] < p30_brightness_low) |
        (df['brightness'] > p30_brightness_high) |
        (df['detection_score'] < p30_det_score)
    ) & (df['face_count'] == 1)
    
    df.loc[bad_mask, 'quality_category'] = 'bad'
    
    return df

def test_threshold_combination(df, thresholds):
    """Test a specific threshold combination"""
    min_face_size, min_blur, min_brightness, max_brightness, min_det_score = thresholds
    
    tp = fp = tn = fn = 0
    
    for _, row in df.iterrows():
        actual_good = row['quality_category'] == 'good'
        
        predicted_good = (
            row['face_count'] == 1 and
            row['face_size'] >= min_face_size and
            row['blur_score'] >= min_blur and
            row['brightness'] >= min_brightness and
            row['brightness'] <= max_brightness and
            row['detection_score'] >= min_det_score
        )
        
        if actual_good and predicted_good:
            tp += 1
        elif not actual_good and predicted_good:
            fp += 1
        elif not actual_good and not predicted_good:
            tn += 1
        else:  # actual_good and not predicted_good
            fn += 1
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score, tp, fp, tn, fn

def main():
    # Dataset path
    dataset_dir = Path('../face-id-project/images_dataset')
    
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return
    
    # Collect all person directories
    person_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith('ST')])
    
    print(f"Found {len(person_dirs)} people in dataset")
    print(f"Analyzing all photos...\n")
    
    # Analyze all photos
    all_metrics = []
    total_images = 0
    
    for person_dir in person_dirs:
        images = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))
        total_images += len(images)
    
    print(f"Total images to process: {total_images}")
    print("This will take approximately 10-15 minutes...\n")
    
    with tqdm(total=total_images, desc="Processing") as pbar:
        for person_dir in person_dirs:
            images = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))
            
            for img_path in images:
                metrics = analyze_photo(img_path)
                if metrics:
                    all_metrics.append(metrics)
                pbar.update(1)
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    print(f"\n✓ Processed {len(df)} images")
    print(f"  - With faces: {len(df[df['face_count'] > 0])}")
    print(f"  - No face: {len(df[df['face_count'] == 0])}")
    print(f"  - Multiple faces: {len(df[df['face_count'] > 1])}")
    
    # Auto-categorize
    df = categorize_quality(df)
    
    # Save raw metrics
    df.to_csv('dataset_metrics.csv', index=False)
    print(f"\n✓ Raw metrics saved to dataset_metrics.csv")
    
    print("\n" + "="*60)
    print("QUALITY DISTRIBUTION")
    print("="*60)
    print(df['quality_category'].value_counts())
    
    # Statistics
    good_photos = df[df['quality_category'] == 'good']
    bad_photos = df[df['quality_category'].isin(['bad', 'no_face', 'multiple_faces'])]
    
    print("\n" + "="*60)
    print("METRIC STATISTICS")
    print("="*60)
    
    if len(good_photos) > 0:
        print("\nGood photos:")
        print(f"  Face size:       {good_photos['face_size'].mean():.1f} ± {good_photos['face_size'].std():.1f}")
        print(f"  Blur score:      {good_photos['blur_score'].mean():.1f} ± {good_photos['blur_score'].std():.1f}")
        print(f"  Brightness:      {good_photos['brightness'].mean():.1f} ± {good_photos['brightness'].std():.1f}")
        print(f"  Detection score: {good_photos['detection_score'].mean():.3f} ± {good_photos['detection_score'].std():.3f}")
    
    bad_with_faces = bad_photos[bad_photos['face_count'] == 1]
    if len(bad_with_faces) > 0:
        print("\nBad photos (with face):")
        print(f"  Face size:       {bad_with_faces['face_size'].mean():.1f} ± {bad_with_faces['face_size'].std():.1f}")
        print(f"  Blur score:      {bad_with_faces['blur_score'].mean():.1f} ± {bad_with_faces['blur_score'].std():.1f}")
        print(f"  Brightness:      {bad_with_faces['brightness'].mean():.1f} ± {bad_with_faces['brightness'].std():.1f}")
        print(f"  Detection score: {bad_with_faces['detection_score'].mean():.3f} ± {bad_with_faces['detection_score'].std():.3f}")
    
    # Test threshold combinations
    print("\n" + "="*60)
    print("TESTING THRESHOLD COMBINATIONS")
    print("="*60 + "\n")
    
    # Generate threshold configs based on data percentiles
    p40 = good_photos.quantile(0.40)
    p50 = good_photos.quantile(0.50)
    p60 = good_photos.quantile(0.60)
    
    threshold_configs = [
        # Very loose (high recall)
        (int(p40['face_size'] * 0.7), int(p40['blur_score'] * 0.7), 30, 230, 0.6),
        # Loose
        (int(p40['face_size'] * 0.8), int(p40['blur_score'] * 0.8), 40, 220, 0.7),
        # Medium-loose
        (int(p40['face_size']), int(p40['blur_score']), 50, 210, 0.75),
        # Medium (RECOMMENDED START)
        (int(p50['face_size']), int(p50['blur_score']), 55, 205, 0.80),
        # Medium-strict
        (int(p60['face_size']), int(p60['blur_score']), 60, 200, 0.85),
        # Strict
        (int(p60['face_size'] * 1.1), int(p60['blur_score'] * 1.1), 65, 195, 0.88),
        # Very strict (high precision)
        (int(p60['face_size'] * 1.2), int(p60['blur_score'] * 1.2), 70, 190, 0.90),
    ]
    
    results = []
    
    for idx, thresholds in enumerate(threshold_configs, 1):
        min_face_size, min_blur, min_brightness, max_brightness, min_det_score = thresholds
        
        accuracy, precision, recall, f1_score, tp, fp, tn, fn = test_threshold_combination(df, thresholds)
        
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
        
        print(f"Config {idx}: Acc={accuracy:.1%}, P={precision:.2f}, R={recall:.2f}, F1={f1_score:.3f}, FP={fp}, FN={fn}")
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('threshold_test_results.csv', index=False)
    print(f"\n✓ Results saved to threshold_test_results.csv")
    
    # Find best configuration (balanced F1)
    best_config = df_results.loc[df_results['f1_score'].idxmax()]
    
    print("\n" + "="*60)
    print("RECOMMENDED THRESHOLDS (Best F1 Score)")
    print("="*60)
    print(f"MIN_FACE_SIZE = {int(best_config['min_face_size'])}")
    print(f"MIN_BLUR = {int(best_config['min_blur'])}")
    print(f"MIN_BRIGHTNESS = {int(best_config['min_brightness'])}")
    print(f"MAX_BRIGHTNESS = {int(best_config['max_brightness'])}")
    print(f"MIN_DET_SCORE = {best_config['min_det_score']:.2f}")
    print(f"\nPERFORMANCE:")
    print(f"  Accuracy:  {best_config['accuracy']:.1%}")
    print(f"  Precision: {best_config['precision']:.2f} (% of accepted photos are truly good)")
    print(f"  Recall:    {best_config['recall']:.2f} (% of good photos are accepted)")
    print(f"  F1 Score:  {best_config['f1_score']:.3f}")
    print(f"\nERRORS:")
    print(f"  False Positives: {int(best_config['false_positive'])} (bad photos incorrectly accepted)")
    print(f"  False Negatives: {int(best_config['false_negative'])} (good photos incorrectly rejected)")
    print("="*60)
    
    # Save thresholds as JSON for easy import
    recommended_thresholds = {
        'MIN_FACE_SIZE': int(best_config['min_face_size']),
        'MIN_BLUR': int(best_config['min_blur']),
        'MIN_BRIGHTNESS': int(best_config['min_brightness']),
        'MAX_BRIGHTNESS': int(best_config['max_brightness']),
        'MIN_DET_SCORE': float(best_config['min_det_score']),
        'performance': {
            'accuracy': float(best_config['accuracy']),
            'precision': float(best_config['precision']),
            'recall': float(best_config['recall']),
            'f1_score': float(best_config['f1_score'])
        }
    }
    
    with open('recommended_thresholds.json', 'w') as f:
        json.dump(recommended_thresholds, f, indent=2)
    
    print(f"\n✓ Thresholds saved to recommended_thresholds.json")
    print("\nNext step: Apply these thresholds to face_extraction_app.py")

if __name__ == '__main__':
    main()
