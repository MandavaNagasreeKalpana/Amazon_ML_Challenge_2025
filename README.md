# Amazon_ML_Challenge_2025

# 🧠 Smart Product Pricing - ML Challenge 2025

This repository contains the complete solution for the **Unstop ML Challenge 2025: Smart Product Pricing**. We developed a multi-modal machine learning pipeline to predict e-commerce product prices by holistically analyzing textual descriptions and product images. Our final model achieved a validation SMAPE score of **53.65%**, a significant improvement from our baseline of 66.76%, by implementing a robust feature engineering and modeling workflow.

## ✨ Key Features

Our approach is built on three pillars: sophisticated feature engineering, multi-modal feature generation, and robust ensemble modeling.

  * **Advanced Text Parsing:** A custom, iterative text parser (up to `v6`) was built using regular expressions to accurately extract critical structured data like `brand`, `num_of_packs`, `net_qty`, and `unit` from the raw `catalog_content`. Our error analysis proved this was the most critical step for improving model accuracy.

  * **Multi-Modal Embeddings:** We leveraged both visual and textual data by generating deep feature embeddings:

      * 👁️ **Image Embeddings:** A pre-trained `MobileNetV3-Small` model was used to convert each product image into a 576-dimension vector, capturing visual attributes like color, shape, and packaging style.
      * 📝 **Text Embeddings:** The `all-MiniLM-L6-v2` model from `sentence-transformers` was used to convert each product description into a 384-dimension vector, capturing its semantic meaning and context.

  * **Ensemble Modeling:** To create a more robust and accurate final prediction, we trained an ensemble of three specialized **LightGBM (`LGBMRegressor`)** models:

    1.  A **Text Specialist** trained on structured and text features.
    2.  A **Vision Specialist** trained on structured and image features.
    3.  A **Hybrid Model** trained on all features combined.
        The final price is an average of the predictions from these three experts.

  * **Deployment Ready:** The project includes a complete inference pipeline, encapsulated in a final interactive script, which can load all trained models and preprocessing rules to make live predictions on new, unseen products.

## 🛠️ Tech Stack

  * **Data Manipulation & Preprocessing:** `pandas`, `numpy`, `scikit-learn`
  * **Modeling:** `LightGBM`
  * **Hyperparameter Tuning:** `Optuna`
  * **Embeddings:** `PyTorch`, `torchvision`, `sentence-transformers`
  * **Deployment & Serialization:** `joblib`, interactive Python scripts

## 📁 Repository Structure

```
.
├── models/                     # Contains saved preprocessing objects
│   ├── deployment_models/      # Contains the three final trained .pkl model files
│   ├── scaler_v2.pkl
│   └── outlier_caps_v2.npy
│
├── dataset/                    # All raw, intermediate, and final .csv files
│   ├── train.csv
│   ├── test.csv
│   ├── train_final_cleaned_v2.csv
│   └── test_final_cleaned_v2.csv
│
├── Ml_train_preprocessing.ipynb # Notebook for the end-to-end training data pipeline
├── test_preprocessing.ipynb    # Notebook for the end-to-end test data pipeline
├── ML_model_challenge.ipynb    # Notebook for model training and inference
└── README.md                   # This file
```

## 🚀 How to Run

1.  **Preprocessing:** Execute the `Ml_train_preprocessing.ipynb` notebook to generate the cleaned training data (`train_final_cleaned_v2.csv`) and save the necessary preprocessing objects (`scaler_v2.pkl`, etc.). Then, run the `test_preprocessing.ipynb` notebook to prepare the test data.

2.  **Training:** Run the final model training script within the `ML_model_challenge.ipynb` notebook. This will train the three specialist models on the full, cleaned dataset and save them to the `models/deployment_models/` folder.

3.  **Inference:** Use the final "Interactive Prediction" script in `ML_model_challenge.ipynb` to load the trained models and make live predictions on new product images and descriptions.
