import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import tarfile

# Extract Pascal VOC 2007 dataset
with tarfile.open('VOCtrainval_06-Nov-2007.tar', 'r') as tar:
    tar.extractall()

with tarfile.open('VOCtest_06-Nov-2007.tar', 'r') as tar:
    tar.extractall()

# Load dataset
data_dir = './VOC2007'

ds, ds_info = tfds.load(
    'voc/2007',
    data_dir=data_dir,
    split='train[:10%]',
    with_info=True,
    shuffle_files=True,
    download=False
)

print(ds_info)

def preprocess_voc_sample(example):
    image = tf.image.resize(example['image'], (256, 256)).numpy()
    h, w = image.shape[:2]
    boxes = example['objects']['bbox'].numpy()
    labels = [obj['label'].numpy() for obj in example['objects']]
    abs_boxes = [
        [int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)]
        for ymin, xmin, ymax, xmax in boxes
    ]
    return image.astype(np.uint8), abs_boxes, labels

def generate_sliding_windows(image, step=64, size=(128, 128)):
    h, w, _ = image.shape
    return [
        (x, y, x + size[0], y + size[1])
        for y in range(0, h - size[1], step)
        for x in range(0, w - size[0], step)
    ]

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(areaA + areaB - inter_area)

vgg = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

def prepare_training_data(image, gt_boxes, iou_thresh=0.5):
    X, y = [], []
    for window in generate_sliding_windows(image):
        ious = [compute_iou(window, gt) for gt in gt_boxes]
        label = 1 if max(ious, default=0) >= iou_thresh else 0
        patch = image[window[1]:window[3], window[0]:window[2]]
        if patch.size == 0:
            continue
        resized = cv2.resize(patch, (224, 224))
        input_tensor = preprocess_input(np.expand_dims(resized, axis=0))
        features = vgg.predict(input_tensor)
        X.append(features.flatten())
        y.append(label)
    return np.array(X), np.array(y)

X_train, y_train = [], []
for example in tfds.as_numpy(ds):
    image, boxes, _ = preprocess_voc_sample(example)
    x, y = prepare_training_data(image, boxes)
    X_train.extend(x)
    y_train.extend(y)

clf = LinearSVC()
clf.fit(X_train, y_train)

def non_max_suppression(boxes, scores, threshold=0.3):
    boxes, scores = np.array(boxes), np.array(scores)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= threshold]
    return keep

sample = next(iter(tfds.as_numpy(ds)))
test_image, gt_boxes, _ = preprocess_voc_sample(sample)

features, boxes = [], []
for window in generate_sliding_windows(test_image):
    patch = test_image[window[1]:window[3], window[0]:window[2]]
    if patch.size == 0:
        continue
    resized = cv2.resize(patch, (224, 224))
    input_tensor = preprocess_input(np.expand_dims(resized, axis=0))
    feature = vgg.predict(input_tensor)
    features.append(feature.flatten())
    boxes.append(window)

scores = clf.decision_function(features)
keep_indices = non_max_suppression(boxes, scores)

for i in keep_indices:
    x1, y1, x2, y2 = boxes[i]
    label = "object" if clf.predict([features[i]])[0] == 1 else "background"
    cv2.rectangle(test_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(test_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

plt.figure(figsize=(10, 8))
plt.imshow(test_image)
plt.axis("off")
plt.title("R-CNN Detection with NMS")
plt.show()
