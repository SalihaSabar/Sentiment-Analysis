import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import pickle
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================================
# BI-LSTM MODEL ARCHITECTURE
# ============================================================================

class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMSentiment, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load both trained models"""
    models = {}
    
    # Load TF-IDF + Logistic Regression model
    try:
        tfidf_vectorizer = joblib.load('weights/tf-idf+logreg/tfidf_vectorizer.pkl')
        tfidf_model = joblib.load('weights/tf-idf+logreg/tfidf_logreg_model.pkl')
        models['tfidf_vectorizer'] = tfidf_vectorizer
        models['tfidf_model'] = tfidf_model
        print("✅ TF-IDF + Logistic Regression model loaded successfully!")
        print(f"   Vectorizer features: {tfidf_vectorizer.get_feature_names_out().shape[0]:,}")
        print(f"   Model classes: {tfidf_model.classes_}")
    except FileNotFoundError as e:
        print(f"❌ Error loading TF-IDF model: {e}")
        print("   Please run 'python tf-idf+logreg.py' first to train the model")
        models['tfidf_vectorizer'] = None
        models['tfidf_model'] = None
    
    # Load BiLSTM + GloVe model
    try:
        # Load vocabulary
        with open('weights/bilstm+glove/bilstm_vocab.pkl', 'rb') as f:
            bilstm_vocab = pickle.load(f)
        
        # Create model instance
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bilstm_model = BiLSTMSentiment(
            vocab_size=len(bilstm_vocab),
            embedding_dim=100,
            hidden_dim=128,
            num_layers=2,
            num_classes=3
        ).to(device)
        
        # Load trained weights
        bilstm_model.load_state_dict(torch.load('weights/bilstm+glove/bilstm_glove_model.pth', map_location=device))
        bilstm_model.eval()
        
        models['bilstm_model'] = bilstm_model
        models['bilstm_vocab'] = bilstm_vocab
        models['device'] = device
        
        print("✅ BiLSTM + GloVe model loaded successfully!")
        print(f"   Vocabulary size: {len(bilstm_vocab):,}")
        print(f"   Device: {device}")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading BiLSTM model: {e}")
        print("   Please run 'python bilstm_glove.py' first to train the model")
        models['bilstm_model'] = None
        models['bilstm_vocab'] = None
        models['device'] = None
    
    return models

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_tfidf(text, vectorizer, model):
    """Predict sentiment using TF-IDF + Logistic Regression"""
    if vectorizer is None or model is None:
        return None, None, None
    
    # Vectorize text
    text_vectorized = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    confidence = np.max(probabilities)
    
    # Map prediction to sentiment
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_map[prediction]
    
    return sentiment, confidence, probabilities

def text_to_sequence(text, vocab, max_length=200):
    """Convert text to sequence of indices"""
    words = text.lower().split()
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words[:max_length]]
    return torch.tensor(sequence, dtype=torch.long)

def predict_bilstm(text, model, vocab, device):
    """Predict sentiment using BiLSTM + GloVe"""
    if model is None or vocab is None:
        return None, None, None
    
    model.eval()
    
    # Preprocess text
    sequence = text_to_sequence(text, vocab)
    sequence = sequence.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(sequence)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = torch.max(probabilities).item()
    
    # Map prediction to sentiment
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_map[predicted_class]
    
    return sentiment, confidence, probabilities.cpu().numpy()[0]

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_sentiment(text, models):
    """Analyze sentiment of text using both models"""
    print(f"\n🔍 Analyzing: \"{text}\"\n")
    print("-" * 60)
    
    results = {}
    
    # TF-IDF + Logistic Regression
    if models['tfidf_model'] is not None:
        sentiment, confidence, probabilities = predict_tfidf(
            text, models['tfidf_vectorizer'], models['tfidf_model']
        )
        results['TF-IDF'] = {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': probabilities
        }
        print(f"🤖 TF-IDF + Logistic Regression:")
        print(f"   Sentiment: {sentiment}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Probabilities: [Neg: {probabilities[0]:.3f}, Neu: {probabilities[1]:.3f}, Pos: {probabilities[2]:.3f}]")
    else:
        print("❌ TF-IDF model not available")
    
    print()
    
    # BiLSTM + GloVe
    if models['bilstm_model'] is not None:
        sentiment, confidence, probabilities = predict_bilstm(
            text, models['bilstm_model'], models['bilstm_vocab'], models['device']
        )
        results['BiLSTM'] = {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': probabilities
        }
        print(f"🧠 BiLSTM + GloVe:")
        print(f"   Sentiment: {sentiment}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Probabilities: [Neg: {probabilities[0]:.3f}, Neu: {probabilities[1]:.3f}, Pos: {probabilities[2]:.3f}]")
    else:
        print("❌ BiLSTM model not available")
    
    print("-" * 60)
    
    # Compare results if both models are available
    if len(results) == 2:
        print("\n📊 Model Comparison:")
        tfidf_sent = results['TF-IDF']['sentiment']
        bilstm_sent = results['BiLSTM']['sentiment']
        
        if tfidf_sent == bilstm_sent:
            print(f"   ✅ Both models agree: {tfidf_sent}")
        else:
            print(f"   ⚠️  Models disagree: TF-IDF={tfidf_sent}, BiLSTM={bilstm_sent}")
        
        tfidf_conf = results['TF-IDF']['confidence']
        bilstm_conf = results['BiLSTM']['confidence']
        print(f"   Confidence: TF-IDF={tfidf_conf:.3f}, BiLSTM={bilstm_conf:.3f}")
    
    return results

def batch_test(texts, models):
    """Test multiple texts and compare model performance"""
    results = []
    
    for text in texts:
        result = {'text': text}
        
        # TF-IDF prediction
        if models['tfidf_model'] is not None:
            sentiment, confidence, _ = predict_tfidf(text, models['tfidf_vectorizer'], models['tfidf_model'])
            result['tfidf_sentiment'] = sentiment
            result['tfidf_confidence'] = confidence
        
        # BiLSTM prediction
        if models['bilstm_model'] is not None:
            sentiment, confidence, _ = predict_bilstm(text, models['bilstm_model'], models['bilstm_vocab'], models['device'])
            result['bilstm_sentiment'] = sentiment
            result['bilstm_confidence'] = confidence
        
        results.append(result)
    
    return pd.DataFrame(results)

def visualize_comparison(batch_results):
    """Visualize model comparison results"""
    if len(batch_results) == 0 or 'tfidf_sentiment' not in batch_results.columns or 'bilstm_sentiment' not in batch_results.columns:
        print("❌ Not enough data for visualization")
        return
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sentiment distribution comparison
    tfidf_counts = batch_results['tfidf_sentiment'].value_counts()
    bilstm_counts = batch_results['bilstm_sentiment'].value_counts()
    
    x = np.arange(len(tfidf_counts))
    width = 0.35
    
    ax1.bar(x - width/2, tfidf_counts.values, width, label='TF-IDF + Logistic Regression', alpha=0.8)
    ax1.bar(x + width/2, bilstm_counts.values, width, label='BiLSTM + GloVe', alpha=0.8)
    
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Count')
    ax1.set_title('Sentiment Distribution Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tfidf_counts.index)
    ax1.legend()
    
    # Confidence comparison
    ax2.scatter(batch_results['tfidf_confidence'], batch_results['bilstm_confidence'], alpha=0.7)
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Diagonal line
    ax2.set_xlabel('TF-IDF Confidence')
    ax2.set_ylabel('BiLSTM Confidence')
    ax2.set_title('Confidence Comparison')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Agreement analysis
    agreement = (batch_results['tfidf_sentiment'] == batch_results['bilstm_sentiment']).sum()
    total = len(batch_results)
    print(f"\n📊 Model Agreement Analysis:")
    print(f"   Agreement: {agreement}/{total} ({agreement/total*100:.1f}%)")
    print(f"   Disagreement: {total-agreement}/{total} ({(total-agreement)/total*100:.1f}%)")

# ============================================================================
# INTERACTIVE TESTING
# ============================================================================

def interactive_testing(models):
    """Interactive interface for testing sentiment analysis"""
    print("\n🎯 Interactive Sentiment Analysis Testing")
    print("Enter 'quit' to exit\n")
    
    while True:
        text = input("Enter text to analyze: ")
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if text.strip():
            analyze_sentiment(text, models)
        else:
            print("Please enter some text to analyze.")
        
        print()

def test_sample_texts(models):
    """Test both models on sample texts"""
    sample_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The food was okay, nothing special but not bad either.",
        "This is the worst experience I've ever had. Terrible service!",
        "The product works as expected, good value for money.",
        "I'm really disappointed with the quality. Waste of money."
    ]
    
    print("📝 Testing both models on sample texts:\n")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"Text {i}: {text}")
        
        # TF-IDF prediction
        if models['tfidf_model'] is not None:
            tfidf_sentiment, tfidf_conf, tfidf_probs = predict_tfidf(text, models['tfidf_vectorizer'], models['tfidf_model'])
            print(f"   TF-IDF: {tfidf_sentiment} (confidence: {tfidf_conf:.3f})")
        
        # BiLSTM prediction
        if models['bilstm_model'] is not None:
            bilstm_sentiment, bilstm_conf, bilstm_probs = predict_bilstm(text, models['bilstm_model'], models['bilstm_vocab'], models['device'])
            print(f"   BiLSTM: {bilstm_sentiment} (confidence: {bilstm_conf:.3f})")
        
        print()

def test_batch_texts(models):
    """Test with a variety of texts"""
    test_texts = [
        "I love this product! It's exactly what I needed.",
        "The service was terrible and the food was cold.",
        "It's okay, nothing special but gets the job done.",
        "Absolutely brilliant! Can't recommend it enough.",
        "Waste of money, completely disappointed.",
        "The quality is good and the price is reasonable.",
        "Horrible experience, would never buy again.",
        "Pretty decent, meets my expectations."
    ]
    
    batch_results = batch_test(test_texts, models)
    print("📋 Batch Test Results:")
    print(batch_results.to_string(index=False))
    
    return batch_results

def print_model_summary(models):
    """Print model performance summary"""
    print("\n📈 Model Performance Summary\n")
    
    if models['tfidf_model'] is not None:
        print("🤖 TF-IDF + Logistic Regression:")
        print(f"   Model type: {type(models['tfidf_model']).__name__}")
        print(f"   Features: {models['tfidf_vectorizer'].get_feature_names_out().shape[0]:,}")
        print(f"   Classes: {models['tfidf_model'].classes_.tolist()}")
        print()
    
    if models['bilstm_model'] is not None:
        print("🧠 BiLSTM + GloVe:")
        print(f"   Model type: {type(models['bilstm_model']).__name__}")
        print(f"   Vocabulary size: {len(models['bilstm_vocab']):,}")
        print(f"   Parameters: {sum(p.numel() for p in models['bilstm_model'].parameters()):,}")
        print(f"   Device: {models['device']}")
        print()
    
    print("✅ Both models are ready for sentiment analysis!")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run the sentiment analysis testing"""
    print("🚀 Sentiment Analysis Model Testing")
    print("=" * 50)
    
    # Load models
    print("\n📦 Loading models...")
    models = load_models()
    
    if models['tfidf_model'] is None and models['bilstm_model'] is None:
        print("\n❌ No models available. Please train the models first:")
        print("   python tf-idf+logreg.py")
        print("   python bilstm_glove.py")
        return
    
    # Print model summary
    print_model_summary(models)
    
    # Test sample texts
    print("\n" + "=" * 50)
    test_sample_texts(models)
    
    # Test batch texts
    print("\n" + "=" * 50)
    batch_results = test_batch_texts(models)
    
    # Visualize results
    if len(batch_results) > 0:
        print("\n" + "=" * 50)
        visualize_comparison(batch_results)
    
    # Interactive testing
    print("\n" + "=" * 50)
    print("Would you like to test your own texts? (y/n): ", end="")
    choice = input().lower().strip()
    
    if choice in ['y', 'yes']:
        interactive_testing(models)
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main() 