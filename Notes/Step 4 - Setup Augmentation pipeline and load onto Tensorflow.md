# 5. Running Augmentation Pipeline Over Dataset Partitions (Custom Code)

## 5.1 Objective

In this step, we:

1. Apply our **Albumentations augmentation pipeline** to **every image** in the `train`, `test`, and `val` partitions.
    
2. Generate **60 augmented versions** per base image.
    
3. Store the results in an organized `aug_data` directory containing new images and YOLO-formatted labels.
    
4. Load these augmented images into TensorFlow as **scaled datasets** (ready for training).
    

---

## 5.2 Loop Through Partitions and Apply Augmentation

```python
for partition in ['train','test','val']:
    for image in (Path('data') / partition / 'images').iterdir():
        img = cv2.imread(str(image))

        if img is None:
            print(f"Failed to read image : {image}")
            continue

        label_path = Path('data') / partition / 'labels' / f'{image.stem}.txt'
        bboxes = [[0.0005, 0.0005, 0.001, 0.001]]
        class_labels = [0]

        if label_path.exists():
            bboxes, class_labels = load_yolo_labels(label_path)
            print(f'{image.stem} ==> ',[class_labels[0]+1], bboxes)
        else:
            with open(Path('data') / partition / 'labels' /  f'{image.stem}.txt','w') as F:
                F.write("0 0.0005 0.0005 0.001 0.001\n")
                print(f'{image.stem} ==> ',class_labels, bboxes)
```

### Explanation:

- We iterate over all images in each of the three partitions.
    
- Attempt to load the image using OpenCV. If loading fails, skip.
    
- Try to load corresponding YOLO labels from a `.txt` file.
    
    - If labels are missing, assign a **default bounding box** and save it.
        
    - This avoids crashing the pipeline when labels are missing.
        

---

## 5.3 Perform 60 Augmentations Per Image

```python
        try:
            for x in range(60):
                augmented = augmentor(image = img,
                                      bboxes = bboxes,
                                      class_labels=class_labels)

                aug_image_path = Path('aug_data') / partition / 'images' / f'{image.stem}.{x}.jpg'
                cv2.imwrite(str(aug_image_path), augmented['image'])

                aug_label_path = Path('aug_data') / partition / 'labels' / f'{image.stem}.{x}.txt'

                with open(aug_label_path, 'w') as f:
                    if augmented['bboxes']:
                        for i, bbox in enumerate(augmented['bboxes']):
                            class_id = 1
                            x_centre, y_centre, width, height = bbox
                            f.write(f"{class_id} {x_centre} {y_centre} {width} {height} \n")
                    else:
                        f.write("0 0.0005 0.0005 0.001 0.001\n")
        except Exception as e:
            print(f"Error augmenting {image.name} : {e}")
```

### Explanation:

- Each image is augmented **60 times** using the predefined `augmentor`.
    
- For each augmented version:
    
    - The new image is saved to `aug_data/{partition}/images/`.
        
    - The corresponding YOLO-formatted label is saved to `aug_data/{partition}/labels/`.
        
- If augmentation fails (e.g., due to bad input), it prints an error.
    

---

## 5.4 Load Augmented Images into TensorFlow Pipeline

```python
train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
train_images = train_images.map(lambda x: x/255)

test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
test_images = test_images.map(lambda x: x/255)

val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
val_images = val_images.map(lambda x: x/255)
```

### Explanation:

- Images are loaded using `tf.data.Dataset.list_files()`.
    
- We use `map()` three times:
    
    1. `load_image`: Decodes the image.
        
    2. `resize`: Resizes it to 120x120 for faster training.
        
    3. `/255`: Scales pixel values to [0, 1] range (useful for sigmoid activations).
        
- `shuffle=False` ensures that images are loaded in the same order as their labels.
    

---

## ✅ Summary

|Step|Action|
|---|---|
|✅|Applied augmentation to every image in `train`, `test`, and `val`|
|✅|Generated 60 augmented images per original sample|
|✅|Saved new images and YOLO labels to organized folders in `aug_data`|
|✅|Built TensorFlow pipelines for each partition, ready for model training|

You now have a fully prepared and augmented dataset that can be efficiently passed to a deep learning model!