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
- Florence Tushabe, Samuel Mwesige, Kasule Vicent, Emmanuel Othieno, Sarah Musani, and team  
- Michigan Medicine — for commissioning and supporting this project  
- Kaggle dataset contributors  
- Soroti University and Kumi Hospital for image collection
=======
