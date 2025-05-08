
# R-CNN with VOC 2007

This repository contains a simplified implementation of the seminal R-CNN (Regions with Convolutional Neural Networks) paper — “Rich feature hierarchies for accurate object detection and semantic segmentation” — using the Pascal VOC 2007 dataset. It demonstrates the core ideas of the R-CNN pipeline, including region proposals, deep feature extraction using VGG16, classification with a Linear SVM, and post-processing via Non-Maximum Suppression (NMS).

## 🔗 Original Paper

Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014).  
📄 [Rich feature hierarchies for accurate object detection and semantic segmentation (CVPR 2014)](https://arxiv.org/abs/1311.2524)

## 📌 Key Highlights

- ✅ Uses TensorFlow Datasets to load a subset of Pascal VOC 2007.
- ✅ Extracts region proposals using sliding windows (simplified Selective Search).
- ✅ Extracts features via a pre-trained VGG16 model (ImageNet weights).
- ✅ Classifies regions with a Linear SVM trained on extracted features.
- ✅ Applies Non-Maximum Suppression to reduce redundant detections.
- ✅ End-to-end visualization of predicted bounding boxes.

## 🧠 Architecture Overview

```
Input Image
    ↓
Region Proposals (sliding window)
    ↓
Resized Proposals → VGG16 (CNN Feature Extractor)
    ↓
Flattened Features → Linear SVM (Classifier)
    ↓
Predicted Scores
    ↓
Non-Maximum Suppression
    ↓
Final Object Detections
```

## 📦 Dependencies

Install the required Python libraries:
```bash
pip install opencv-python-headless tensorflow scikit-learn matplotlib tensorflow-datasets
```

## 📂 Dataset

This implementation uses only a 10% subset of the Pascal VOC 2007 dataset for training, for faster experimentation:
```python
ds, ds_info = tfds.load('voc/2007', split='train[:10%]', with_info=True, shuffle_files=True)
```

## 🔬 Seminal Importance

The R-CNN model was a transformative milestone in computer vision, bridging classical vision methods with deep learning. It introduced the idea of:
- Using CNNs to extract rich, hierarchical features from region proposals.
- Training class-specific SVMs using deep features.
- A modular object detection pipeline that set the foundation for Fast R-CNN, Faster R-CNN, and Mask R-CNN.

This architecture significantly improved object detection accuracy over traditional methods (like DPMs) and laid the groundwork for future real-time and end-to-end object detectors.

## 📊 Results

The notebook provides visualization of detection results on sample test images, showing bounding boxes labeled as “object” or “background”.

## 🚀 Future Enhancements

- Integrate Selective Search for better region proposals.
- Train on the full VOC dataset for improved performance.
- Replace SVM with softmax classifier and finetune the CNN.

---

Made with ❤️ for computer vision research and learning.
