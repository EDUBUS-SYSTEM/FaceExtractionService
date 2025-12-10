"""
Photo Preparation Helper for Face Enrollment
Converts photos to base64 for API testing

Usage:
    python prepare_photos.py <student_folder>
    
Example:
    python prepare_photos.py photos/student_001
    
Output: base64_encoded_photos.json
"""

import os
import sys
import json
import base64
from pathlib import Path

def convert_photo_to_base64(photo_path):
    """Convert a photo file to base64 string"""
    try:
        with open(photo_path, 'rb') as f:
            photo_data = f.read()
            base64_str = base64.b64encode(photo_data).decode('utf-8')
            return base64_str
    except Exception as e:
        print(f"Error converting {photo_path}: {str(e)}")
        return None

def prepare_photos_for_student(folder_path):
    """Prepare all photos in folder for enrollment"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return None
    
    # Supported image formats
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Find all photos (case-insensitive, no duplicates)
    photos = set()  # Use set to avoid duplicates
    for ext in extensions:
        photos.update(folder.glob(f'*{ext}'))
        photos.update(folder.glob(f'*{ext.upper()}'))
    
    photos = list(photos)  # Convert back to list
    
    if len(photos) == 0:
        print(f"❌ No photos found in {folder_path}")
        print(f"   Supported formats: {', '.join(extensions)}")
        return None
    
    print(f"✓ Found {len(photos)} photos in {folder_path}")
    
    # Convert to base64
    base64_photos = []
    for photo_path in sorted(photos):
        print(f"  Processing: {photo_path.name}...")
        b64 = convert_photo_to_base64(photo_path)
        if b64:
            base64_photos.append(b64)
            print(f"    ✓ Converted ({len(b64)} chars)")
    
    return base64_photos

def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_photos.py <student_folder>")
        print("")
        print("Example:")
        print("  python prepare_photos.py photos/student_001")
        return
    
    folder_path = sys.argv[1]
    
    print("="*50)
    print("Face Enrollment - Photo Preparation")
    print("="*50)
    print("")
    
    # Convert photos
    base64_photos = prepare_photos_for_student(folder_path)
    
    if not base64_photos:
        print("\n❌ Failed to prepare photos")
        return
    
    # Create output
    output = {
        'photos': base64_photos,
        'count': len(base64_photos),
        'source_folder': str(folder_path)
    }
    
    # Save to file
    output_file = 'base64_photos.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("")
    print("="*50)
    print(f"✅ SUCCESS! Prepared {len(base64_photos)} photos")
    print(f"📁 Saved to: {output_file}")
    print("="*50)
    print("")
    print("Next steps:")
    print("1. Copy the 'photos' array from base64_photos.json")
    print("2. Use in enrollment API request body:")
    print("   {")
    print('     "studentId": "550E8400-E29B-41D4-A716-446655440010",')
    print('     "facePhotos": [...paste photos array here...]')
    print("   }")

if __name__ == '__main__':
    main()
