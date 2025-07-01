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

- **polars** - Fast DataFrame library for large datasets
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **nltk** - Natural language processing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **wordcloud** - Word cloud generation
- **jupyter** - Interactive notebooks (optional)

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

Details about model training and evaluation coming soon...

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset providers and Kaggle community
- Contributors and maintainers
- Open source ML community
