# Sentiment Analysis Project

A comprehensive sentiment analysis tool that performs sentiment classification on multiple datasets using machine learning techniques.

## 🚀 Features

- **Multi-dataset Support**: Works with various text datasets
- **Multiple ML Models**: Implements different classification algorithms
- **Easy-to-use Interface**: Simple API for sentiment prediction
- **Model Evaluation**: Comprehensive metrics and performance analysis

## 📊 Supported Datasets

- IMDB Movie Reviews
- Twitter Sentiment Analysis
- Amazon Product Reviews
- Yelp Reviews
- Custom datasets

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

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

4. **Verify installation**
   ```bash
   python -c "import pandas, numpy, sklearn; print('All dependencies installed successfully!')"
   ```

### Dependencies

The project requires the following main packages:

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **nltk** - Natural language processing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **jupyter** - Interactive notebooks (optional)

### Environment Variables

Create a `.env` file in the root directory:
