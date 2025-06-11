# 6. Preparing and Combining Labels with Augmented Images for Model Training

In this step, we build a label processing pipeline to load YOLO-format annotations, map them to TensorFlow datasets, and combine them with the corresponding images for model training.

---

## 6.1 Loading YOLO Labels into TensorFlow

We begin by defining a custom label loading function called `new_load_yolo_labels`, which reads YOLO-formatted `.txt` label files:

```python
def new_load_yolo_labels(label_file_path):
    label_file_path = label_file_path.numpy().decode('utf-8')  # Decode tensor to string

    bboxes = []
    class_labels = []

    with open(label_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)

    return (
        np.array(class_labels, dtype=np.uint8),
        np.array(bboxes, dtype=np.float32)
    )
```

- This function parses each line of the YOLO `.txt` file.
    
- Class labels are stored as integers.
    
- Bounding boxes are stored as float arrays (YOLO format: center-x, center-y, width, height).
    

---

## 6.2 Mapping Labels to TensorFlow Datasets

We map our label loading function to TensorFlow datasets for `train`, `test`, and `val` sets:

```python
train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.txt', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(new_load_yolo_labels, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.txt', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(new_load_yolo_labels, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.txt', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(new_load_yolo_labels, [x], [tf.uint8, tf.float16]))
```

- We use `tf.data.Dataset.list_files()` to get all label paths.
    
- We use `tf.py_function()` to wrap our NumPy-based label loader inside a TensorFlow-compatible function.
    

---

## 6.3 Zipping Labels and Images Together

To train our model, we need to pair each image with its corresponding labels:

```python
train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)
```

- We `zip` each image with its annotation.
    
- `.shuffle()` ensures randomness in training.
    
- `.batch(8)` allows 8 samples per batch.
    
- `.prefetch(4)` optimizes input pipeline performance.
    

---

## 6.4 Visualizing Sample Augmented Data

To ensure our pipeline is functioning correctly, we visualize the results:

### a. Fetching a Sample Batch

```python
data_samples = train.as_numpy_iterator()
res = data_samples.next()  # res[0] = images, res[1] = (class_labels, bboxes)
```

### b. Converting YOLO Format to Pixel Corners

```python
def yolo_to_pixelcorner(bbox, img_w, img_h):
    x_center, y_center, w, h = bbox
    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h

    x_min = int(x_center - w / 2)
    y_min = int(y_center - h / 2)
    x_max = int(x_center + w / 2)
    y_max = int(y_center + h / 2)

    return [x_min, y_min, x_max, y_max]
```

### c. Displaying with Matplotlib

```python
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4):
    sample_image = res[0][idx].copy()
    sample_coords = res[1][1][idx].flatten()

    pixel_corners = yolo_to_pixelcorner(sample_coords, 120, 120)

    cv2.rectangle(sample_image, (pixel_corners[0], pixel_corners[1]), (pixel_corners[2], pixel_corners[3]), (25, 0, 0), 2)

    ax[idx].imshow(sample_image)
```

- This displays the first 4 images in a batch with their bounding boxes.
    
- Bounding boxes are converted from YOLO to corner format for drawing.
    

---

## ✅ Summary of Completed Step

- ✅ Defined a label loading function for YOLO format.
    
- ✅ Loaded labels for all image partitions (`train`, `test`, `val`).
    
- ✅ Combined images and labels using `tf.data.Dataset.zip()`.
    
- ✅ Batched, shuffled, and prefetched the data for performance.
    
- ✅ Visualized augmented image samples with drawn bounding boxes.
    

You now have a complete, shuffled, batched, and labeled dataset ready for model training in TensorFlow!