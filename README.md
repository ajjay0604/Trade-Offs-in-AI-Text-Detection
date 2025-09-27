# Trade-Offs in AI-Text Detection: A Comparative Study of DistilBERT, ALBERT, and TinyBERT

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-orange)](https://huggingface.co/transformers)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)


## 📖 About The Project

The rapid evolution of Large Language Models (LLMs) has made their output nearly indistinguishable from human prose, creating significant challenges for academic integrity and the spread of misinformation. While many detection methods rely on large, resource-intensive models, this approach is often impractical for real-world applications.

This research has been documented and accepted for publication in the proceedings of a peer-reviewed conference, highlighting its contribution to the field of AI text detection.

This project provides a systematic, comparative study that focuses on the optimal balance between performance and practicality. We fine-tuned and benchmarked three lightweight models: **DistilBERT, ALBERT, and TinyBERT** against a traditional TF-IDF baseline to guide the development of efficient and deployable AI detection systems.

## 📊 Performance vs Efficiency Graph :

<p align="center">
<img width="1278" height="726" alt="Screenshot 2025-09-27 at 4 23 56 PM" src="https://github.com/user-attachments/assets/e105c72d-2f8f-4a6e-891c-b38eedf46772" />
</p>  

## 🏆 Key Findings

The study reveals that the most efficient models can achieve performance nearly identical to larger, slower models.  
**TinyBERT** offers the most optimal balance:  
- Achieves a near-perfect **F1-Score of 0.9936**  
- Over **4.7x faster** than ALBERT  
- Nearly **3x smaller** than ALBERT  

These results demonstrate that hyper-efficient models are a mature and viable solution for scalable, real-world AI detection.



## 📖 About The Project

[cite_start]The rapid evolution of Large Language Models (LLMs) has made their output nearly indistinguishable from human prose, creating significant challenges for academic integrity and the spread of misinformation[cite: 645, 646, 647]. [cite_start]While many detection methods rely on large, resource-intensive models, this approach is often impractical for real-world applications[cite: 648].

[cite_start]This project provides a systematic, comparative study that shifts the focus from maximizing accuracy at all costs to finding an optimal balance between performance and practicality[cite: 679]. [cite_start]We fine-tuned and benchmarked three prominent lightweight models—**DistilBERT, ALBERT, and TinyBERT**—against a traditional TF-IDF baseline to provide a clear guide for developing efficient and deployable AI detection systems[cite: 649, 650, 654].


## 🚀 Getting Started

Follow these instructions to set up the project locally and reproduce our results.

### 1. Prerequisites

* Python 3.9 or later
* Git and Git LFS (for handling datasets, if needed)

### 2. Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/YOUR_USERNAME/Trade-Offs-in-AI-Text-Detection.git](https://github.com/YOUR_USERNAME/Trade-Offs-in-AI-Text-Detection.git)
    cd Trade-Offs-in-AI-Text-Detection
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

### 3. Dataset Setup

The dataset used for this project is the **[DAIGT V2 Train Dataset](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset)** from Kaggle.

### ### Option 1: Manual Download
1.  Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset).
2.  Download the files.
3.  Place the `train_v2_drcat_02.csv` file inside the `/data` directory in this project.

### ### Option 2: Kaggle API (Recommended)
1.  Install the Kaggle API: `pip install kaggle`
2.  Run the following command to download and place the dataset automatically:
    ```sh
    kaggle datasets download -d thedrcat/daigt-v2-train-dataset -p data/ --unzip
    ```

## 📈 Reproducing the Results

To run the experiments, simply open and execute the cells in the `notebooks/main_experiment.ipynb` notebook.

The notebook is designed to:
1.  [cite_start]Load and preprocess the dataset from the `/data` folder[cite: 755].
2.  [cite_start]Train and evaluate the TF-IDF + Logistic Regression baseline model[cite: 768].
3.  [cite_start]Systematically fine-tune, evaluate, and measure the efficiency (inference time and model size) of DistilBERT, ALBERT, and TinyBERT[cite: 793].
4.  [cite_start]Generate the final comparison table and visualizations seen in the paper[cite: 854].



## 📂 Repository Structure

```
├── data/
│   └── README.md         # Placeholder for dataset
├── notebooks/
│   └── main_experiment.ipynb # Main notebook for all experiments
├── assets/
│   └── performance_graph.png # Your results graph image
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```


## ## 📜 License

This project is distributed under the MIT License. See the `LICENSE` file for more information.
