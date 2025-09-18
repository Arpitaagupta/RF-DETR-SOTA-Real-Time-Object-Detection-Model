# 🚀 RT-DETR Object Detection – Training & Deployment  

This repository contains my work for the **AI R&D Engineer assignment** on **RT-DETR (Real-Time DEtection TRansformer)**. The project demonstrates **end-to-end training, evaluation, model conversion, and inference** using PyTorch and TensorRT with C++ deployment.  

## 📌 **Overview**
 🔹 Implemented **RT-DETR** for object detection.  
 🔹 Trained and fine-tuned the model on a chosen dataset.  
 🔹 Evaluated performance using **mAP** and other metrics.  
 🔹 Converted trained PyTorch model → **ONNX → TensorRT** for optimized inference.  
 🔹 Deployed real-time inference pipeline in **C++** (based on YOLOv8-TensorRT repo).  

## 🛠 **Technologies Used**
- **Deep Learning Frameworks**: PyTorch, TorchVision  
- **Deployment & Optimization**: TensorRT, ONNX  
- **Programming Languages**: Python, C++  
- **Dataset**: COCO / Pascal VOC / Custom dataset  

## 📑 **Milestones**

### 1️⃣ Dataset Preparation
- Downloaded and preprocessed dataset.  
- Converted annotations into COCO format.  
- Split into **train/validation/test sets**.  

---

### 2️⃣ Training
- Trained RT-DETR using provided configs.  
- Adjusted **hyperparameters** (batch size, learning rate, epochs).  
- Saved trained model weights.  

---

### 3️⃣ Evaluation
- Evaluated using **mAP@0.5, mAP@0.5:0.95**.  
- Plotted **precision-recall curves**.  
- Analyzed false positives/negatives for improvements.  

---

### 4️⃣ Model Conversion
- Exported model from **PyTorch → ONNX → TensorRT**.  
- Verified conversion correctness with dummy inputs.  
- Achieved faster inference with TensorRT engine.  

---

### 5️⃣ Inference (C++)
- Implemented **C++ inference** pipeline.  
- Modified YOLOv8-TensorRT repo for RT-DETR.  
- Tested on sample images & videos.  
- Compared inference speed: **TensorRT vs PyTorch**.  

---

## 📊 **Results**
- ✅ Achieved **XX mAP** on validation dataset.  
- ✅ Inference speed improved by **X times** with TensorRT.  
- ✅ Successfully deployed inference in **C++** for real-time detection.  

---

## 🚀 **How to Run**

### 🔹 Training
```bash
python tools/train.py -c configs/rtdetr_r50vd_6x_coco.yml
```

---

### 🔹 Evaluation
```
python tools/eval.py -c configs/rtdetr_r50vd_6x_coco.yml -o weights/model_final.pth
```
- Evaluated using **mAP@0.5, mAP@0.5:0.95**.  
- Plotted **precision-recall curves**.  
- Analyzed false positives/negatives for improvements.  

---

### 🔹 Model Conversion
```
python tools/export_onnx.py --weights weights/model_final.pth
trtexec --onnx=model.onnx --saveEngine=model.engine
```
- Exported model from **PyTorch → ONNX → TensorRT**.  
- Verified conversion correctness with dummy inputs.  
- Achieved faster inference with TensorRT engine.  

---

### 🔹 Inference (C++)
```
cd inference_cpp
mkdir build && cd build
cmake ..
make
./rtdetr_infer ../samples/input.jpg
```
- Implemented **C++ inference** pipeline.  
- Modified YOLOv8-TensorRT repo for RT-DETR.  
- Tested on sample images & videos.  
- Compared inference speed: **TensorRT vs PyTorch**.  

---

## 📊 **Results**
- ✅ Achieved **XX mAP** on validation dataset.  
- ✅ Inference speed improved by **X times** with TensorRT.  
- ✅ Successfully deployed inference in **C++** for real-time detection.  



## 🖼 **Sample Output**
| Input Image | Detection Result |
|-------------|------------------|
| ![input](results/input.jpg) | ![output](results/output.jpg) |

---

## 🚀 **How to Run**

### 🔹 Training
```bash
python tools/train.py -c configs/rtdetr_r50vd_6x_coco.yml
```

---
## 📌 References
- [RT-DETR GitHub](https://github.com/lyuwenyu/RT-DETR)  
- [YOLOv8-TensorRT GitHub](https://github.com/triple-Mu/YOLOv8-TensorRT)  
- [NVIDIA TensorRT Documentation](https://developer.nvidia.com/tensorrt)  
