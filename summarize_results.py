import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('dataset_metrics.csv')

print("="*60)
print("DATASET ANALYSIS SUMMARY")
print("="*60)
print(f"\nTotal images processed: {len(df)}")
print(f"  - With faces detected: {len(df[df['face_count'] > 0])}")
print(f"  - No face detected: {len(df[df['face_count'] == 0])}")
print(f"  - Multiple faces: {len(df[df['face_count'] > 1])}")

# Analyze photos with single face
single_face = df[df['face_count'] == 1]

print(f"\n" + "="*60)
print("QUALITY METRICS (Photos with single face)")
print("="*60)
print(single_face[['face_size', 'blur_score', 'brightness', 'detection_score']].describe())

# Calculate recommended thresholds (30th percentile)
print(f"\n" + "="*60)
print("RECOMMENDED THRESHOLDS (30th Percentile)")
print("="*60)

p30_face_size = single_face['face_size'].quantile(0.30)
p30_blur = single_face['blur_score'].quantile(0.30)
p30_brightness_low = single_face['brightness'].quantile(0.15)
p30_brightness_high = single_face['brightness'].quantile(0.85)
p30_det_score = single_face['detection_score'].quantile(0.30)

print(f"MIN_FACE_SIZE = {int(p30_face_size)}")
print(f"MIN_BLUR = {int(p30_blur)}")
print(f"MIN_BRIGHTNESS = {int(p30_brightness_low)}")
print(f"MAX_BRIGHTNESS = {int(p30_brightness_high)}")
print(f"MIN_DET_SCORE = {p30_det_score:.2f}")

# Also show median (50th percentile) - recommended for production
print(f"\n" + "="*60)
print("PRODUCTION THRESHOLDS (50th Percentile - RECOMMENDED)")
print("="*60)

p50_face_size = single_face['face_size'].quantile(0.50)
p50_blur = single_face['blur_score'].quantile(0.50)
p50_brightness_low = single_face['brightness'].quantile(0.25)
p50_brightness_high = single_face['brightness'].quantile(0.75)
p50_det_score = single_face['detection_score'].quantile(0.50)

print(f"MIN_FACE_SIZE = {int(p50_face_size)}")
print(f"MIN_BLUR = {int(p50_blur)}")
print(f"MIN_BRIGHTNESS = {int(p50_brightness_low)}")
print(f"MAX_BRIGHTNESS = {int(p50_brightness_high)}")
print(f"MIN_DET_SCORE = {p50_det_score:.2f}")

# Save to JSON
import json

recommended = {
    "conservative_30th_percentile": {
        "MIN_FACE_SIZE": int(p30_face_size),
        "MIN_BLUR": int(p30_blur),
        "MIN_BRIGHTNESS": int(p30_brightness_low),
        "MAX_BRIGHTNESS": int(p30_brightness_high),
        "MIN_DET_SCORE": float(p30_det_score),
        "description": "Rejects bottom 30% quality - More strict"
    },
    "recommended_50th_percentile": {
        "MIN_FACE_SIZE": int(p50_face_size),
        "MIN_BLUR": int(p50_blur),
        "MIN_BRIGHTNESS": int(p50_brightness_low),
        "MAX_BRIGHTNESS": int(p50_brightness_high),
        "MIN_DET_SCORE": float(p50_det_score),
        "description": "Balanced threshold - RECOMMENDED for production"
    }
}

with open('recommended_thresholds.json', 'w') as f:
    json.dump(recommended, f, indent=2)

print(f"\n✅ Thresholds saved to recommended_thresholds.json")
