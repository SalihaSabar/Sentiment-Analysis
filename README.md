# Sentiment Analysis Project

A comprehensive sentiment analysis tool that performs sentiment classification on multiple datasets using machine learning techniques.

## 🚀 Features

- **Multi-dataset Support**: Works with various text datasets
- **Multiple ML Models**: Implements different classification algorithms
- **Easy-to-use Interface**: Simple API for sentiment prediction
- **Model Evaluation**: Comprehensive metrics and performance analysis
- **Large-scale Processing**: Optimized for handling large datasets (6GB+ of text data)

## 📊 Datasets

The project uses several large-scale datasets for sentiment analysis:

### Core Datasets

1. **IMDB Movie Reviews**

   - 50,000 movie reviews labeled by sentiment
   - [Dataset Source](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - Features: Binary sentiment (Positive/Negative)
   - Size: 50K reviews

2. **X/Twitter Sentiment140**

   - 1.6 million tweets labeled by sentiment
   - [Dataset Source](https://www.kaggle.com/datasets/kazanova/sentiment140/data)
   - Features: Binary sentiment (Positive/Negative)
   - Size: 1.6M tweets

3. **Reddit Sentiment**

   - Combined Twitter and Reddit posts with sentiment labels
   - [Dataset Source](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset)
   - Features: Multi-class sentiment
   - Size: Varies

4. **Yelp Reviews**
   - Business reviews with star ratings and text
   - [Dataset Source](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset?select=yelp_academic_dataset_review.json)
   - Features: 5-star rating system
   - Size: Multiple millions of reviews

### Dataset Processing

- All datasets are preprocessed for consistency
- Text cleaning and normalization applied
- Combined into a unified format
- Balanced sampling for training

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- Minimum 32GB RAM recommended for full dataset processing
- 10GB free disk space for datasets

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/SalihaSabar/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```

2. **Create a virtual environment**

   ```bash
   # Using venv (recommended)
   python -m venv venv

   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download datasets**

   ```bash
   # Create directories for datasets
   mkdir -p raw_datasets processed_datasets

   # Download datasets manually from the following sources and place in raw_datasets/ directory:
   # - IMDB: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
   # - Yelp: https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset
   # - Twitter: https://www.kaggle.com/datasets/kazanova/sentiment140
   # - Reddit: https://www.kaggle.com/datasets/arnavsharmaas/reddit-sentiment-analysis
   #
   # Extract and rename files to:
   # - raw_datasets/IMDB.csv
   # - raw_datasets/Yelp.json
   # - raw_datasets/Twitter.csv
   # - raw_datasets/Reddit.csv
   ```

5. **Verify installation**
   ```bash
   python -c "import pandas, numpy, sklearn; print('All dependencies installed successfully!')"
   ```

### Dependencies

The project requires the following main packages:

#### Core Dependencies

- **polars** - Fast DataFrame library for large datasets
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **torch** - Deep learning framework for BiLSTM model
- **nltk** - Natural language processing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **wordcloud** - Word cloud generation
- **jupyter** - Interactive notebooks (optional)

#### Transformer Dependencies (for BERT/RoBERTa)

- **transformers** - Hugging Face transformers library
- **datasets** - Dataset processing utilities
- **accelerate** - Faster training utilities
- **tokenizers** - Fast tokenization

Install transformer dependencies:

```bash
pip install -r requirements_transformers.txt
```

## 📈 Data Analysis

The project includes comprehensive exploratory data analysis (EDA):

- Text length distributions
- Word frequency analysis
- Sentiment distribution
- Word clouds
- Text preprocessing statistics

Run the EDA script:

```bash
python eda.py
```

Results will be saved in the `eda_plots/` directory.

## 🤖 Model Training

The project uses a two-step approach for model training and supports multiple architectures:

### 1. Automatic Dataset Labeling

We use a state-of-the-art transformer model ([cardiffnlp/twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)) to automatically label the preprocessed combined dataset. This is done using the `label_dataset.py` script, which assigns sentiment labels (Negative, Neutral, Positive) to each text entry.

Run the labeling script:

```bash
python label_dataset.py
```

This will create a new file `processed_datasets/combined_dataset_with_labels.csv` containing the original data and the predicted sentiment labels.

### 2. Training Models

We implement multiple approaches for sentiment classification:

**Available Models:**

1. **TF-IDF + Logistic Regression** - Traditional ML approach
2. **BiLSTM + GloVe** - Deep learning with word embeddings
3. **🚀 Transformer Models** - State-of-the-art BERT/RoBERTa (NEW!)

---

#### A. TF-IDF + Logistic Regression (scikit-learn)

A traditional machine learning approach using TF-IDF vectorization and Logistic Regression classification.

Run the training script:

```bash
python tf-idf+logreg.py
```

Steps:

1. Load the labeled dataset (`processed_datasets/combined_dataset_with_labels.csv`).
2. Split the data into training and test sets.
3. Fit a TF-IDF vectorizer on the training text.
4. Train a Logistic Regression model with hyperparameter tuning.
5. Evaluate the model on the test set using standard metrics (accuracy, F1, etc.).

#### B. BiLSTM + GloVe Embeddings (PyTorch)

A deep learning approach using Bidirectional LSTM with pre-trained GloVe word embeddings.

**Prerequisites:**

- Download GloVe embeddings from [Stanford NLP](https://nlp.stanford.edu/projects/glove/)
- Place `glove.6B.100d.txt` in the project root directory

Run the training script:

```bash
python bilstm_glove.py
```

Steps:

1. Load the labeled dataset and GloVe embeddings.
2. Build vocabulary and tokenize text sequences.
3. Create embedding matrix using GloVe vectors.
4. Build and train a BiLSTM model with PyTorch.
5. Evaluate the model and save for inference.

**Model Architecture:**

- Embedding layer with GloVe weights (100d)
- Bidirectional LSTM layers (2 layers, 128 hidden units)
- Dense layers with dropout for regularization
- Softmax output for 3-class classification

**Features:**

- Automatic GPU/CPU detection for PyTorch training
- GPU-accelerated data processing with cuDF (optional)
- Early stopping to prevent overfitting
- Learning rate scheduling
- Custom dataset and collate functions for variable-length sequences

#### C. 🚀 **NEW: Transformer Fine-tuning (BERT/RoBERTa)**

State-of-the-art transfer learning approach using pre-trained transformer models fine-tuned on your custom dataset.

**Supported Models:**

- BERT (base/large)
- RoBERTa (base/large)
- DistilBERT
- Twitter-RoBERTa (pre-trained on social media data)

**Install transformer dependencies:**

```bash
pip install -r requirements_transformers.txt
```

**Quick Start:**

```bash
# Train BERT and RoBERTa models
python transformer_finetuning.py

# Or use configuration management
python transformer_config.py
```

**Advanced Configuration:**

The transformer training system provides flexible configuration options:

```python
from transformer_config import build_config

# Build configuration for specific use case
config = build_config(
    model_type="roberta-base",
    training_mode="production",
    custom_overrides={"num_epochs": 4, "batch_size": 32}
)
```

**Features:**

- 🎯 **Multiple Model Support**: BERT, RoBERTa, DistilBERT, and more
- ⚡ **GPU Optimization**: Mixed precision training, gradient accumulation
- 📊 **Smart Data Loading**: Efficient chunk-based loading for large datasets
- 🎨 **Advanced Training**: Early stopping, learning rate scheduling, class weighting
- 📈 **Comprehensive Evaluation**: Detailed metrics, confusion matrices, confidence analysis
- 💾 **Easy Deployment**: Automatic model saving with inference examples
- ⚙️ **Hardware Adaptive**: Automatic batch size and memory optimization

**Model Architectures:**

| Model        | Parameters | Speed  | Performance | Use Case         |
| ------------ | ---------- | ------ | ----------- | ---------------- |
| DistilBERT   | 66M        | ⚡⚡⚡ | ⭐⭐⭐      | Fast inference   |
| BERT Base    | 110M       | ⚡⚡   | ⭐⭐⭐⭐    | Balanced         |
| RoBERTa Base | 125M       | ⚡⚡   | ⭐⭐⭐⭐⭐  | Best overall     |
| BERT Large   | 340M       | ⚡     | ⭐⭐⭐⭐⭐  | Maximum accuracy |

**Training Process:**

1. **Data Preprocessing**: Automatic tokenization and encoding
2. **Model Loading**: Pre-trained weights from Hugging Face Hub
3. **Fine-tuning**: Task-specific training with advanced techniques
4. **Evaluation**: Comprehensive metrics and visualization
5. **Deployment**: Ready-to-use models with inference scripts

**Example Usage:**

```python
# Quick inference after training
from transformer_inference import SentimentPredictor

predictor = SentimentPredictor("weights/transformers/roberta-base/final_model")
result = predictor.predict("This movie was fantastic!")
print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.3f})")
```

## 🧪 Model Testing & Comparison

Test and compare all trained models using the integrated testing system:

```bash
python test_models.py
```

**Features:**

- 🔄 **Multi-Model Loading**: Automatically loads all available trained models
- 📊 **Side-by-Side Comparison**: Compare predictions across all models
- 📈 **Confidence Analysis**: Identify the most confident predictions
- 🎯 **Interactive Testing**: Test your own texts in real-time
- 📋 **Batch Processing**: Test multiple texts efficiently
- 📊 **Visual Analytics**: Generate comparison charts and metrics

**Supported Models:**

- ✅ TF-IDF + Logistic Regression
- ✅ BiLSTM + GloVe Embeddings
- ✅ BERT Transformer (if trained)
- ✅ RoBERTa Transformer (if trained)
- ✅ Any other fine-tuned transformers

**Example Output:**

```
🔍 Analyzing: "This movie was absolutely fantastic!"

🤖 TF-IDF + Logistic Regression:
   Sentiment: Positive
   Confidence: 0.856
   Probabilities: [Neg: 0.089, Neu: 0.055, Pos: 0.856]

🧠 BiLSTM + GloVe:
   Sentiment: Positive
   Confidence: 0.923
   Probabilities: [Neg: 0.034, Neu: 0.043, Pos: 0.923]

🤖 BERT Transformer:
   Sentiment: Positive
   Confidence: 0.967
   Probabilities: [Neg: 0.012, Neu: 0.021, Pos: 0.967]

🤖 RoBERTa Transformer:
   Sentiment: Positive
   Confidence: 0.982
   Probabilities: [Neg: 0.008, Neu: 0.010, Pos: 0.982]

📊 Model Comparison:
   ✅ All models agree: Positive
   🏆 Most confident: RoBERTa (0.982)
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset providers and Kaggle community
- Contributors and maintainers
- Open source ML community
