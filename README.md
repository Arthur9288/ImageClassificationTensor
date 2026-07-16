# AmazonImageClassification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.93%2B-00a393)
![Google Earth Engine](https://img.shields.io/badge/Google_Earth_Engine-API-green)

AmazonImageClassification is a machine learning and computer vision project designed to detect deforestation patterns in the Amazon rainforest using satellite imagery.

The project features an automated data collection pipeline, a trained deep learning model for image classification, and a web interface for real-time analysis.

<img width="1913" height="1013" alt="Image" src="https://github.com/user-attachments/assets/a035b6b4-ef71-4966-86ae-cf23f67a6d97" />

<img width="1912" height="893" alt="Image" src="https://github.com/user-attachments/assets/31c4ccb6-23af-4e8b-bd05-2a3b86c7c81a" />

## Features

- **Automated Data Collection:** Integrated with the Google Earth Engine API to download Sentinel-2 satellite images based on geographic coordinates. Includes an active learning script that auto-classifies new images to expand the dataset.
- **Image Classification Model:** Utilizes Transfer Learning (MobileNetV2) and Data Augmentation techniques to classify images into two categories: "Desmatamento" (Deforestation) and "Floresta Intacta" (Intact Forest).
- **REST API:** A FastAPI-based microservice that handles model inference and serves the web application.
- **Web Interface:** A frontend built with Vanilla JS, CSS, and Leaflet.js, allowing users to draw a bounding box over a map and receive real-time classification results.

## Prerequisites

- Python 3.9+
- Google Cloud Platform account (for Google Earth Engine API access)

## Installation

1. Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/your-username/AmazonImageClassification.git
cd AmazonImageClassification
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file based on the provided template and add your Google Cloud credentials.
```bash
cp .env.example .env
```

## Usage

### 1. Training the Model
To train the model on your local dataset (placed in `data/Desmatamento` and `data/Floresta_Intacta`), run:
```bash
python train.py
```

### 2. Starting the API and Web Interface
To start the FastAPI server and serve the frontend application:
```bash
python main.py
```
Access the application by navigating to `http://localhost:8000` in your web browser. API documentation (Swagger) is available at `http://localhost:8000/docs`.

### 3. Data Collection

<img width="771" height="756" alt="Image" src="https://github.com/user-attachments/assets/3532889e-9f8e-4d09-a7eb-0e7478aadbdd" />

To download and automatically classify new satellite images for dataset expansion:
```bash
python collect_images.py
```

## Docker Deployment

To build and run the application using Docker:

```bash
docker build -t amazon-image-classification .
docker run -p 8000:8000 amazon-image-classification
```

## Legacy Implementation

The original implementation of this project relied on a Jupyter Notebook using `pyautogui` to automate the browser and capture screenshots from Google Earth. This approach has been completely deprecated in favor of the current architecture.

Legacy execution screenshots for historical reference:
<img src="https://github.com/user-attachments/assets/0d8394c5-9e12-4af0-b48e-9370421f2ff7" width="600">
<img src="https://github.com/user-attachments/assets/15c28faf-8014-4c34-8f33-cd70c60392a8" width="600">
