# AgroKD-Net 🌱
### Lightweight Plant Segmentation Using Knowledge Distillation

AgroKD-Net is a computer vision project focused on binary plant segmentation. The goal was to train a smaller segmentation model that could learn from a larger, more capable model while using fewer parameters.

I built a teacher–student pipeline to explore whether **knowledge distillation could transfer useful segmentation knowledge from a DeepLabV3-ResNet50 teacher to a MobileNetV3-Large student**.

The project includes a teacher model, a baseline student trained without distillation, and a second student trained using knowledge distillation. Comparing all three helped me understand the trade-offs between segmentation quality, model size, and inference speed.
 
---

## 📌 Project Overview

Semantic segmentation assigns a class to every pixel in an image. Unlike image classification, which predicts one label for an entire image, segmentation identifies the regions belonging to each class.

For this project, I used binary masks to distinguish plant pixels from the background.

The main objectives were:

- Train a strong teacher segmentation model.
- Build a lightweight student model.
- Compare student performance with and without knowledge distillation.
- Evaluate segmentation quality using IoU, mean IoU, and pixel accuracy.
- Compare model parameters, checkpoint sizes, and inference latency.

**Note:** This implementation treats the dataset masks as binary plant/background labels. It does not currently distinguish crops from weeds.

## 🧠 Model Architecture

The project uses three models for comparison.

### 1. Teacher — DeepLabV3-ResNet50

DeepLabV3 with a ResNet50 backbone serves as the teacher model. It learns to predict a class for every pixel and provides the student with soft probability distributions during distillation.

The teacher is trained using ground-truth masks and serves as the higher-capacity reference model.

### 2. Baseline Student — MobileNetV3-Large

The student uses MobileNetV3-Large as its feature extractor, followed by a lightweight convolutional decoder.

The decoder converts the extracted features into two segmentation channels: background and plant. Its output is resized to match the input image dimensions.

This model is trained directly against the ground-truth masks, without knowledge distillation.

### 3. KD Student — MobileNetV3-Large

The distilled student uses the same architecture as the baseline student.

The difference is in the training process: it learns from both the ground-truth labels and the teacher's output probabilities. The idea is to provide additional information about how the teacher distinguishes the two classes.

The student architecture remains unchanged, so knowledge distillation does not automatically reduce its parameter count or inference cost compared with the baseline.

## ⚙️ How It Works

The overall workflow is:

1. **Load the dataset:** Read the RGB images and their corresponding binary masks.
2. **Preprocess the data:** Resize images and masks to a consistent resolution and normalize the images.
3. **Train the teacher:** Train DeepLabV3-ResNet50 using the ground-truth segmentation masks.
4. **Train the baseline student:** Train MobileNetV3-Large using the standard segmentation loss.
5. **Train the KD student:** Train another MobileNetV3-Large using ground-truth supervision combined with teacher predictions.
6. **Evaluate all models:** Compare their performance on the same held-out test set.
7. **Visualize predictions:** Inspect the original images, ground-truth masks, and predictions from all three models.
8. **Benchmark efficiency:** Measure parameter counts, checkpoint sizes, and inference latency.

### Preprocessing

- Image size: 256 × 192 pixels (width × height)
- Image channels: RGB
- Image normalization: ImageNet mean and standard deviation
- Mask resizing: Nearest-neighbor interpolation
- Number of segmentation classes: 2
- Training batch size: 8

Images are resized and normalized before entering the models. Masks are resized without introducing intermediate label values.

## 🛠️ Tech Stack

- **Language:** Python
- **Deep learning:** PyTorch
- **Computer vision:** Torchvision
- **Dataset handling:** Hugging Face Datasets
- **Experiment environment:** Google Colab
- **Evaluation and visualization:** NumPy, Pandas, Matplotlib

## 🎓 Training and Knowledge Distillation

The models were trained for eight epochs each using AdamW.

| Setting | Value |
|---|---|
| Teacher architecture | DeepLabV3-ResNet50 |
| Student architecture | MobileNetV3-Large |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Ground-truth loss | Cross-entropy |
| KD temperature | 4 |
| KD loss weight (alpha) | 0.5 |

The teacher and baseline student use the standard cross-entropy segmentation loss.

For the KD student, the training objective combines the ground-truth loss with a KL-divergence loss between the teacher's and student's temperature-scaled output distributions.

The distillation loss is normalized across pixels before being combined with the ground-truth loss. This avoids the excessively large loss values observed in an earlier version of the experiment.

In simplified form:

`Total Loss = (1 - alpha) × Ground-Truth Loss + alpha × Distillation Loss`

The temperature controls how much the class probability distributions are softened during distillation.


*Built as an experimental computer vision project using PyTorch and Google Colab.*
