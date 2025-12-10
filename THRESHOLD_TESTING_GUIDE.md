# 🧪 Photo Quality Threshold Testing Guide

## 📋 Chuẩn bị (10-15 phút)

### Bước 1: Tạo thư mục test data
```bash
cd face-extraction-service
mkdir test_data
mkdir test_data/good
mkdir test_data/bad
```

### Bước 2: Phân loại ảnh của 50 người

**Ảnh GOOD (Chất lượng tốt):**
```
test_data/good/
  ├── person01_photo1.jpg  ✅ Rõ nét, ánh sáng tốt
  ├── person01_photo2.jpg  ✅ Nhìn thẳng camera
  ├── person02_photo1.jpg  ✅ Không bị che
  └── ...
```

**Ảnh BAD (Chất lượng kém):**
```
test_data/bad/
  ├── person01_blurry.jpg  ❌ Bị mờ (chuyển động)
  ├── person02_dark.jpg    ❌ Quá tối
  ├── person03_covered.jpg ❌ Tay che mặt
  ├── person04_far.jpg     ❌ Đứng xa (mặt nhỏ)
  └── ...
```

**Gợi ý phân loại:**
- Mỗi người: 3-5 ảnh
- 50 người × 4 ảnh = **200 ảnh**
- Phân loại: **70% good, 30% bad** (140 good, 60 bad)

---

## ▶️ Chạy Test (5-10 phút)

### Bước 3: Cài package (nếu chưa có)
```bash
pip install pandas opencv-python insightface
```

### Bước 4: Chạy script
```bash
python test_thresholds.py
```

**Output mẫu:**
```
Loading InsightFace model...
✓ Model loaded

Found 140 good images
Found 60 bad images

Analyzing good quality photos...
Analyzing bad quality photos...

============================================================
METRIC STATISTICS
============================================================

Good photos metrics:
  Face size:       156.3 ± 42.1
  Blur score:      89.2 ± 28.5
  Brightness:      124.6 ± 18.9
  Detection score: 0.951 ± 0.034

Bad photos metrics:
  Face size:       78.4 ± 35.2
  Blur score:      35.8 ± 22.1
  Brightness:      98.3 ± 45.2
  Detection score: 0.723 ± 0.158

============================================================
TESTING THRESHOLD COMBINATIONS
============================================================

Config 1: Accuracy=78.5%, F1=0.821, FP=15, FN=28
Config 2: Accuracy=84.0%, F1=0.878, FP=10, FN=22
Config 3: Accuracy=91.5%, F1=0.932, FP=6, FN=11  ← Best!
Config 4: Accuracy=87.0%, F1=0.901, FP=8, FN=18
Config 5: Accuracy=82.5%, F1=0.865, FP=12, FN=23

✓ Results saved to threshold_results.csv

============================================================
RECOMMENDED THRESHOLDS (Best F1 Score)
============================================================
MIN_FACE_SIZE = 80
MIN_BLUR = 60
MIN_BRIGHTNESS = 50
MAX_BRIGHTNESS = 210
MIN_DET_SCORE = 0.75

Accuracy: 91.5%
False Positives (bad photos accepted): 6
False Negatives (good photos rejected): 11
============================================================
```

---

## 📊 Phân tích kết quả (10-15 phút)

### Bước 5: Mở file `threshold_results.csv`
```csv
config,min_face_size,min_blur,accuracy,f1_score,false_positive,false_negative
1,60,40,0.785,0.821,15,28
2,70,50,0.840,0.878,10,22
3,80,60,0.915,0.932,6,11  ← Best
4,90,70,0.870,0.901,8,18
```

### Bước 6: Chọn config phù hợp

**Ưu tiên Accuracy cao:**
- Chọn config có `accuracy` cao nhất
- VD: Config 3 (91.5%)

**Ưu tiên ít False Negative (không reject ảnh tốt):**
- Chọn config có `false_negative` thấp
- VD: Config 1 hoặc 2

**Quyết định:**
```python
# Sửa trong face_extraction_app.py
MIN_FACE_SIZE = 80      # Từ config 3
MIN_BLUR_THRESHOLD = 60
MIN_BRIGHTNESS = 50
MAX_BRIGHTNESS = 210
MIN_DET_SCORE = 0.75
```

---

## ✅ Apply vào Production

### Bước 7: Update code trong `face_extraction_app.py`

```python
# Add validation logic
if face_width < 80:  # Use tested threshold
    return jsonify({'error': 'Face too small'}), 400

laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
if laplacian_var < 60:  # Use tested threshold
    return jsonify({'error': 'Image too blurry'}), 400

brightness = np.mean(face_crop)
if brightness < 50 or brightness > 210:  # Use tested threshold
    return jsonify({'error': 'Poor lighting'}), 400
```

### Bước 8: Test với Mobile App
- Submit ảnh từ Mobile
- Kiểm tra xem có reject đúng không
- Thu thập feedback từ user thực tế

---

## 🔄 Monitoring & Adjustment

### Sau 1 tuần sử dụng:

**Check rejection rate:**
```sql
SELECT 
    rejection_reason, 
    COUNT(*) as count,
    COUNT(*) * 100.0 / (SELECT COUNT(*) FROM enrollment_logs) as percentage
FROM enrollment_logs 
WHERE status = 'rejected'
GROUP BY rejection_reason
```

**Nếu thấy:**
```
blurry: 45% (Quá cao!)
→ Hạ threshold xuống: MIN_BLUR = 50
```

```
too_dark: 2% (OK)
→ Giữ nguyên: MIN_BRIGHTNESS = 50
```

---

## 📈 Kết quả mong đợi

Với 50 người (200 ảnh):
- **Phân loại manual:** 15-20 phút
- **Chạy script:** 5-10 phút
- **Phân tích:** 10-15 phút
- **Apply code:** 10 phút

**Tổng thời gian: ~1 giờ** 🎯

**Output:**
- ✅ Threshold tối ưu cho production
- ✅ Accuracy ước tính ~85-95%
- ✅ Biết rõ trade-off (FP vs FN)
