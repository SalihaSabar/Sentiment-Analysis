import polars as pl
import pandas as pd
from pandarallel import pandarallel

# Download NLTK data once at the start
def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded."""
    import nltk
    print("Checking and downloading NLTK data...")
    required_data = [
        ('tokenizers/punkt', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    
    for data_path, data_name in required_data:
        try:
            nltk.data.find(data_path)
            print(f"✓ {data_name} already downloaded")
        except LookupError:
            print(f"Downloading {data_name}...")
            nltk.download(data_name, quiet=True)
    print("NLTK data ready!")

class TextPreprocessor:
    def __init__(self):
        # Import all required modules
        import re
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        
        # Store modules as instance variables
        self.re = re
        self.word_tokenize = word_tokenize
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def remove_emojis(self, text):
        emoji = self.re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
                        "]+", self.re.UNICODE)
        return self.re.sub(emoji, '', text)
    
    def remove_urls(self, text):
        """Remove URLs from text."""
        url_pattern = r'https?://\S+|www\.\S+'
        return self.re.sub(url_pattern, '', text)
    
    def remove_html_tags(self, text):
        """Remove HTML tags from text."""
        html_pattern = r'<.*?>'
        return self.re.sub(html_pattern, '', text)
    
    def remove_special_chars(self, text):
        """Remove special characters and numbers, keeping only letters and spaces."""
        return self.re.sub(r'[^a-zA-Z\s]', '', text)
    
    def normalize_whitespace(self, text):
        """Normalize whitespace by removing extra spaces."""
        return ' '.join(text.split())
    
    def lemmatize_text(self, text):
        """Lemmatize text to get root words."""
        words = self.word_tokenize(text.lower())
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])
    
    def remove_stopwords(self, text):
        """Remove stop words from text."""
        words = text.split()
        return ' '.join([word for word in words if word.lower() not in self.stop_words])
    
    def preprocess_text(self, text):
        """Apply all preprocessing steps to text."""
        if not isinstance(text, str):
            return ''
            
        text = self.remove_emojis(text)
        text = self.remove_urls(text)
        text = self.remove_html_tags(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)
        text = self.lemmatize_text(text)
        text = self.remove_stopwords(text)
        return text.strip().lower()

def process_with_pandarallel(texts):
    """Process texts using pandarallel."""
    # Initialize pandarallel with progress bar
    pandarallel.initialize(progress_bar=True, verbose=0)
    
    # Create a pandas Series from texts
    series = pd.Series(texts)
    
    # Create preprocessor instance
    preprocessor = TextPreprocessor()
    
    # Process texts in parallel with progress bar
    print("\nPreprocessing texts in parallel...")
    processed_series = series.parallel_apply(preprocessor.preprocess_text)
    
    return processed_series.tolist()

if __name__ == "__main__":
    # Ensure NLTK data is downloaded
    ensure_nltk_data()
    
    print("\nLoading dataset...")
    # Read the dataset using Polars for efficient loading
    df = pl.scan_csv('processed_datasets/combined_dataset.csv')
    
    # Fetch text column and convert to list
    print("\nFetching text data...")
    texts = df.select('text').collect()['text'].to_list()
    
    # Process texts using pandarallel
    preprocessed_texts = process_with_pandarallel(texts)
    
    # Create final dataframe
    print("\nCreating final dataset...")
    result_df = pl.DataFrame({
        'text': texts,
        'preprocessed_text': preprocessed_texts,
        'sentiment': df.select('sentiment').collect()['sentiment'].to_list()
    })
    
    print("\nSaving preprocessed dataset...")
    result_df.write_csv('processed_datasets/combined_dataset_preprocessed.csv')
    
    print("\nSample of preprocessed data:")
    print(result_df.head())
