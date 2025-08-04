<<<<<<< HEAD
# Sickle Cell Disease Detection Using Deep Learning

## 🧬 Project Overview
This project applies deep learning to automate the detection of Sickle Cell Disease (SCD) from microscopic blood smear images, aiming to improve diagnostic accessibility in low-resource settings.

## 🎯 Project Goals
- Develop a deep learning model to classify red blood cells as healthy or sickled
- Achieve high accuracy in detection while maintaining interpretability
- Create a solution that can be deployed in resource-constrained environments

## 🗂️ Project Structure
```
sickle_cell_detection/
├── config/               # Configuration files
├── data/                 # Dataset (not versioned)
│   ├── raw/             # Original data
│   └── processed/       # Processed data
├── notebooks/           # Jupyter notebooks for EDA and prototyping
├── reports/             # Generated analysis and figures
│   └── figures/
├── src/                 # Source code
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model architectures
│   └── utils/          # Utility functions
└── requirements.txt    # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset
The dataset consists of microscopic blood smear images from Uganda, available on Kaggle. It contains:
- Positive (sickled) cells: 844 images
- Negative (healthy) cells: 147 images

## 🤖 Model Development
We experiment with:
- Custom CNN architectures
- Transfer learning with pre-trained models (MobileNetV2, ResNet50)
- Data augmentation techniques
- Class imbalance handling

## 📈 Results
[Summary of model performance and key findings]

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Michigan Medicine for commissioning this project
- Kaggle for hosting the dataset
- Open-source community for valuable tools and libraries
=======
# 🩸 Sickle Cell Disease Detection from Microscopic Images

A deep learning project using TensorFlow/Keras to classify red blood cell images for early detection of Sickle Cell Disease (SCD). This model is trained on a real-world dataset collected from clinical sites in Uganda and supports research in medical imaging, diagnostic assistance, and health equity.

---

## 🔬 Project Overview

This project was **commissioned by Michigan Medicine** to explore the feasibility of applying deep learning to automate sickle cell disease (SCD) detection in both clinical and low-resource settings. SCD is a life-threatening genetic condition that disproportionately affects populations in sub-Saharan Africa and underserved regions globally. Manual detection from blood smear microscopy is time-consuming and requires trained personnel — a barrier where diagnostic capacity is limited.

To address this, the project leverages Convolutional Neural Networks (CNNs) and transfer learning techniques to analyze and classify **microscopic blood smear images** for early signs of SCD. Built on a Kaggle dataset of real-world cell images from Uganda, the model demonstrates high accuracy in distinguishing sickled from normal red blood cells, supporting scalable and accessible diagnostic workflows.

---

## 🚀 Key Features

✅ Trained on authentic clinical samples from Uganda  
🧠 Achieved high accuracy on a small, imbalanced dataset (~422 positive, ~147 clear negative images)  
🔁 Fine-tuned MobileNetV2 and ResNet using transfer learning  
📈 Preprocessing: resizing, normalization, data augmentation  
📊 Evaluated using confusion matrices, precision, recall, and F1-score  
🌍 Supports diagnostic assistance in low-resource healthcare environments  

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- OpenCV for image processing  
- Scikit-learn for evaluation metrics  
- Transfer Learning: ResNet50, MobileNetV2

---

## 📊 Model Workflow

### 📥 Data Collection

- Source: Kaggle – [Sickle Cell Disease Dataset](https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset)
- Origin: Microscopic blood smear images collected in the Teso region of Eastern Uganda (Kumi Hospital, Soroti Regional Referral Hospital, Soroti University)
- Dataset size:
  - 422 positive (sickle cell) images (some labeled with bounding boxes)
  - 147 clear negative images
  - 122 unclear images (excluded from baseline training)

### 🧪 Preprocessing

- Image resizing to 224×224 pixels
- Normalization to [0, 1] pixel scale
- Data augmentation (rotation, flipping, zoom) to improve generalization

### 🏗️ Model Training

- Custom baseline CNN model
- Fine-tuned pretrained models: **MobileNetV2**, **ResNet50**
- Training techniques: early stopping, checkpointing, learning rate scheduling

### 📈 Evaluation

- Accuracy, Precision, Recall, F1-score
- **Confusion matrices** to visualize performance across classes
- ROC curves (optional)
- Grad-CAM interpretability (future work)

---

## 🏥 Research Relevance

This work contributes to Michigan Medicine’s health equity and global health research goals by developing automated, interpretable, and cost-effective tools to support early diagnosis of SCD. In low-resource environments where diagnostic labs are limited, tools like this can aid healthcare workers by reducing diagnostic delay, increasing throughput, and minimizing error — ultimately supporting better patient outcomes.

---

## 📌 Future Work

- Integrate Grad-CAM for model interpretability  
- Build a **Streamlit web demo** for clinical research or field testing  
- Expand to **multi-class classification** (e.g., sickling severity or cell types)  
- Evaluate model calibration and reliability under noisy inputs

---

## 📜 License & Acknowledgments

This project uses data from:

> **Tushabe et al. (2024–2025)**  
> _“A Dataset of Microscopic Images of Sickle and Normal Red Blood Cells”_  
> Available on Kaggle: [https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset](https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset)

Dataset prepared with funding from the Government of Uganda through Soroti University Research and Innovation Fund (Project RIF/2022/05). Special thanks to all research collaborators and participating institutions.

---

## 🙌 Acknowledgments

- Florence Tushabe, Samuel Mwesige, Kasule Vicent, Emmanuel Othieno, Sarah Musani, and team  
- Michigan Medicine — for commissioning and supporting this project  
- Kaggle dataset contributors  
- Soroti University and Kumi Hospital for image collection

---

## 📂 Repository Structure

```bash
ml-scd-detection/
│
├── Model_Training_and_Evaluation.ipynb   # Full training + confusion matrix analysis
├── gradcam_comparisons.png               # Sample Grad-CAM overlays (optional future use)
├── data/                                 # (If stored locally) preprocessed image folders
├── README.md                             # This file
└── ...
>>>>>>> d6ae7a6ce05b97788934be9cfb9f1c9ae044d26e
# Sickle-Cell-Detector
