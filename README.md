# 🌱 AgroKD-Net: Edge-Optimized Semantic Segmentation via Knowledge Distillation

**AgroKD-Net** is an ultra-lightweight, edge-optimized computer vision pipeline designed for real-time precision agriculture (e.g., drone-based weed detection or robotic pesticide sprayers). 

This project demonstrates how to compress the spatial awareness of a massive foundation model into an ultra-tiny custom architecture capable of running on low-power edge CPUs without sacrificing minority-class detection accuracy.

## 🚀 Key Engineering Achievements

* **Massive Parameter Reduction (99%+):** Engineered a custom `AgroTinyNet` student model using Depthwise Separable Convolutions and U-Net style skip-connections, slashing parameter count from 11M+ (Teacher) to under 10k.
* **Feature-Based Knowledge Distillation:** Utilized Mean Squared Error (MSE) distillation to force the tiny student model to mimic the intermediate spatial attention maps of a frozen DeepLabV3-MobileNetV3 Teacher.
* **Overcoming the "Majority Class Trap":** Developed a custom **Adaptive Imbalance-Aware Pixel Loss (AIPL)** combining Focal and Dice loss to successfully identify rare weed pixels within overwhelmingly soil-heavy field images.
* **MLOps Ready:** Exported the final computational graph to ONNX format for hardware-agnostic Edge deployment (C++, Mobile, IoT).

## 🧠 Architecture Overview

| Component | Choice | Justification |
| :--- | :--- | :--- |
| **Teacher Model** | DeepLabV3 (MobileNetV3-Large) | Provides robust, generalized edge and shape feature extraction. |
| **Student Model** | Custom `AgroTinyNet_v2` | Depthwise Separable Convolutions + Skip Connections for <10k params. |
| **Loss Function** | Custom `AgroImbalanceLoss` | Forces network attention on minority classes (weeds/crops) over background (soil). |
| **Distillation** | Feature-Map MSE | Aligns the spatial understanding of the Student with the Teacher. |


## 💻 Project Structure

```text
├── assets/                  # Visual proofs and loss curves
├── notebooks/               # Jupyter notebooks for exploration and training
├── src/
│   ├── model.py             # AgroTinyNet_v2 architecture
│   ├── loss.py              # Adaptive Imbalance-Aware Loss function
│   └── export.py            # ONNX computational graph export script
├── requirements.txt         # Environment dependencies
└── README.md
