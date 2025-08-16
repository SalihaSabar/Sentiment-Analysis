import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import pickle
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
import os

warnings.filterwarnings('ignore')

# Try to import transformer dependencies
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available. Install with: pip install transformers")

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
# TRANSFORMER MODEL CLASS
# ============================================================================

class TransformerPredictor:
    """Wrapper for transformer models"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_names = ["Negative", "Neutral", "Positive"]
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the transformer model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"❌ Error loading transformer model: {e}")
            self.model = None
            self.tokenizer = None
    
    def predict(self, text):
        """Predict sentiment using transformer model"""
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return None, None, None
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]
            
            sentiment = self.label_names[predicted_class]
            return sentiment, float(confidence), probabilities
            
        except Exception as e:
            print(f"❌ Error in transformer prediction: {e}")
            return None, None, None

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load all trained models"""
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
    
    # Load Transformer models (BERT/RoBERTa)
    transformer_models = {}
    transformer_paths = {
        'BERT': 'weights/transformers/bert-base/final_model',
        'RoBERTa': 'weights/transformers/roberta-base/final_model'
    }
    
    for model_name, model_path in transformer_paths.items():
        if Path(model_path).exists():
            try:
                predictor = TransformerPredictor(model_path)
                if predictor.model is not None:
                    transformer_models[model_name] = predictor
                    print(f"✅ {model_name} transformer model loaded successfully!")
                    print(f"   Model path: {model_path}")
                    print(f"   Device: {predictor.device}")
            except Exception as e:
                print(f"❌ Error loading {model_name} transformer model: {e}")
    
    models['transformers'] = transformer_models
    
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
    """Analyze sentiment of text using all available models"""
    print(f"\n🔍 Analyzing: \"{text}\"\n")
    print("-" * 80)
    
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
    
    print()
    
    # Transformer models
    if models.get('transformers') and len(models['transformers']) > 0:
        for model_name, predictor in models['transformers'].items():
            sentiment, confidence, probabilities = predictor.predict(text)
            if sentiment is not None:
                results[model_name] = {
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'probabilities': probabilities
                }
                print(f"🤖 {model_name} Transformer:")
                print(f"   Sentiment: {sentiment}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Probabilities: [Neg: {probabilities[0]:.3f}, Neu: {probabilities[1]:.3f}, Pos: {probabilities[2]:.3f}]")
                print()
    
    print("-" * 80)
    
    # Compare results if multiple models are available
    if len(results) >= 2:
        print("\n📊 Model Comparison:")
        
        # Get all sentiments
        sentiments = {model: result['sentiment'] for model, result in results.items()}
        confidences = {model: result['confidence'] for model, result in results.items()}
        
        # Check agreement
        unique_sentiments = set(sentiments.values())
        if len(unique_sentiments) == 1:
            print(f"   ✅ All models agree: {list(unique_sentiments)[0]}")
        else:
            print(f"   ⚠️  Models disagree:")
            for model, sentiment in sentiments.items():
                print(f"      {model}: {sentiment}")
        
        # Show confidences
        print(f"   Confidence scores:")
        for model, confidence in confidences.items():
            print(f"      {model}: {confidence:.3f}")
        
        # Find most confident prediction
        most_confident_model = max(confidences.items(), key=lambda x: x[1])
        print(f"   🏆 Most confident: {most_confident_model[0]} ({most_confident_model[1]:.3f})")
    
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
        
        # Transformer predictions
        if models.get('transformers') and len(models['transformers']) > 0:
            for model_name, predictor in models['transformers'].items():
                sentiment, confidence, _ = predictor.predict(text)
                if sentiment is not None:
                    # Use consistent naming convention
                    model_key = model_name.lower().replace('-', '_').replace('_base', '').replace('_large', '')
                    result[f'{model_key}_sentiment'] = sentiment
                    result[f'{model_key}_confidence'] = confidence
        
        results.append(result)
    
    return pd.DataFrame(results)

def visualize_comparison(batch_results):
    """Visualize model comparison results"""
    if len(batch_results) == 0:
        print("❌ Not enough data for visualization")
        return
    
    print(f"📊 Creating visualization for {len(batch_results)} samples...")
    
    # Get available models for visualization
    available_models = []
    if 'tfidf_sentiment' in batch_results.columns:
        available_models.append('TF-IDF')
    if 'bilstm_sentiment' in batch_results.columns:
        available_models.append('BiLSTM')
    
    # Add transformer models
    transformer_models = []
    for col in batch_results.columns:
        if col.endswith('_sentiment') and col not in ['tfidf_sentiment', 'bilstm_sentiment']:
            model_name = col.replace('_sentiment', '').upper()
            transformer_models.append(model_name)
            available_models.append(model_name)
    
    print(f"   Available models: {', '.join(available_models)}")
    
    if len(available_models) < 2:
        print("❌ Need at least 2 models for comparison visualization")
        return
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sentiment distribution comparison
    sentiment_counts = {}
    for model in available_models:
        if model == 'TF-IDF':
            col = 'tfidf_sentiment'
        elif model == 'BiLSTM':
            col = 'bilstm_sentiment'
        else:
            col = f'{model.lower()}_sentiment'
        
        if col in batch_results.columns:
            sentiment_counts[model] = batch_results[col].value_counts()
            print(f"   {model}: {dict(sentiment_counts[model])}")
    
    print(f"   Total models with sentiment data: {len(sentiment_counts)}")
    
    # Create grouped bar chart
    if len(sentiment_counts) > 0:
        try:
            # Ensure all models have the same sentiment classes
            all_sentiments = set()
            for counts in sentiment_counts.values():
                all_sentiments.update(counts.index)
            all_sentiments = sorted(list(all_sentiments))
            
            print(f"   Sentiment classes found: {all_sentiments}")
            
            # Create consistent counts for all models
            consistent_counts = {}
            for model, counts in sentiment_counts.items():
                consistent_counts[model] = []
                for sentiment in all_sentiments:
                    consistent_counts[model].append(counts.get(sentiment, 0))
            
            x = np.arange(len(all_sentiments))
            width = 0.8 / len(consistent_counts)
            
            for i, (model, counts) in enumerate(consistent_counts.items()):
                offset = (i - len(consistent_counts)/2 + 0.5) * width
                ax1.bar(x + offset, counts, width, label=model, alpha=0.8)
            
            ax1.set_xlabel('Sentiment')
            ax1.set_ylabel('Count')
            ax1.set_title('Sentiment Distribution Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(all_sentiments)
            ax1.legend()
            
        except Exception as e:
            print(f"❌ Error creating sentiment distribution chart: {e}")
            # Create a simple fallback chart
            ax1.text(0.5, 0.5, 'Error creating chart', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Sentiment Distribution Comparison (Error)')
    
    # Confidence comparison (compare first two available models)
    if len(available_models) >= 2:
        model1, model2 = available_models[0], available_models[1]
        
        if model1 == 'TF-IDF':
            col1 = 'tfidf_confidence'
        elif model1 == 'BiLSTM':
            col1 = 'bilstm_confidence'
        else:
            col1 = f'{model1.lower()}_confidence'
            
        if model2 == 'TF-IDF':
            col2 = 'tfidf_confidence'
        elif model2 == 'BiLSTM':
            col2 = 'bilstm_confidence'
        else:
            col2 = f'{model2.lower()}_confidence'
        
        if col1 in batch_results.columns and col2 in batch_results.columns:
            try:
                # Check if we have valid confidence values
                conf1 = batch_results[col1].dropna()
                conf2 = batch_results[col2].dropna()
                
                if len(conf1) > 0 and len(conf2) > 0:
                    ax2.scatter(conf1, conf2, alpha=0.7)
                    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Diagonal line
                    ax2.set_xlabel(f'{model1} Confidence')
                    ax2.set_ylabel(f'{model2} Confidence')
                    ax2.set_title('Confidence Comparison')
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'No valid confidence data', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Confidence Comparison (No Data)')
            except Exception as e:
                print(f"❌ Error creating confidence comparison chart: {e}")
                ax2.text(0.5, 0.5, 'Error creating chart', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Confidence Comparison (Error)')
    
    try:
        plt.tight_layout()
        plt.show()
        print("✅ Visualization completed successfully!")
    except Exception as e:
        print(f"❌ Error displaying visualization: {e}")
        print("   This might be due to display issues or matplotlib backend problems")
        # Try to save the plot instead
        try:
            plt.savefig('sentiment_comparison.png', dpi=300, bbox_inches='tight')
            print("   Plot saved as 'sentiment_comparison.png'")
        except Exception as save_error:
            print(f"   Could not save plot: {save_error}")
        finally:
            plt.close()
    
    # Agreement analysis
    print(f"\n📊 Model Agreement Analysis:")
    
    # Calculate agreement between all available models
    if len(available_models) >= 2:
        agreement_matrix = {}
        for i, model1 in enumerate(available_models):
            for j, model2 in enumerate(available_models[i+1:], i+1):
                if model1 == 'TF-IDF':
                    col1 = 'tfidf_sentiment'
                elif model1 == 'BiLSTM':
                    col1 = 'bilstm_sentiment'
                else:
                    col1 = f'{model1.lower()}_sentiment'
                    
                if model2 == 'TF-IDF':
                    col2 = 'tfidf_sentiment'
                elif model2 == 'BiLSTM':
                    col2 = 'bilstm_sentiment'
                else:
                    col2 = f'{model2.lower()}_sentiment'
                
                if col1 in batch_results.columns and col2 in batch_results.columns:
                    agreement = (batch_results[col1] == batch_results[col2]).sum()
                    total = len(batch_results)
                    agreement_pct = agreement/total*100
                    agreement_matrix[f"{model1} vs {model2}"] = agreement_pct
                    print(f"   {model1} vs {model2}: {agreement}/{total} ({agreement_pct:.1f}%)")
        
        # Overall agreement
        if len(agreement_matrix) > 0:
            avg_agreement = sum(agreement_matrix.values()) / len(agreement_matrix)
            print(f"   Average agreement: {avg_agreement:.1f}%")
    else:
        print("   Need at least 2 models for agreement analysis")
    
    # Show transformer-specific insights if available
    if transformer_models:
        print(f"\n🤖 Transformer Model Insights:")
        for model_name in transformer_models:
            col = f'{model_name.lower()}_sentiment'
            if col in batch_results.columns:
                sentiment_dist = batch_results[col].value_counts()
                confidence_col = f'{model_name.lower()}_confidence'
                avg_confidence = batch_results[confidence_col].mean() if confidence_col in batch_results.columns else 0
                print(f"   {model_name}:")
                print(f"      Sentiment distribution: {dict(sentiment_dist)}")
                print(f"      Average confidence: {avg_confidence:.3f}")

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
    """Test all available models on sample texts"""
    sample_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The food was okay, nothing special but not bad either.",
        "This is the worst experience I've ever had. Terrible service!",
        "The product works as expected, good value for money.",
        "I'm really disappointed with the quality. Waste of money."
    ]
    
    print("📝 Testing all available models on sample texts:\n")
    
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
        
        # Transformer predictions
        if models.get('transformers') and len(models['transformers']) > 0:
            for model_name, predictor in models['transformers'].items():
                sentiment, confidence, probabilities = predictor.predict(text)
                if sentiment is not None:
                    print(f"   {model_name}: {sentiment} (confidence: {confidence:.3f})")
        
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
    
    model_count = 0
    
    if models['tfidf_model'] is not None:
        print("🤖 TF-IDF + Logistic Regression:")
        print(f"   Model type: {type(models['tfidf_model']).__name__}")
        print(f"   Features: {models['tfidf_vectorizer'].get_feature_names_out().shape[0]:,}")
        print(f"   Classes: {models['tfidf_model'].classes_.tolist()}")
        print()
        model_count += 1
    
    if models['bilstm_model'] is not None:
        print("🧠 BiLSTM + GloVe:")
        print(f"   Model type: {type(models['bilstm_model']).__name__}")
        print(f"   Vocabulary size: {len(models['bilstm_vocab']):,}")
        print(f"   Parameters: {sum(p.numel() for p in models['bilstm_model'].parameters()):,}")
        print(f"   Device: {models['device']}")
        print()
        model_count += 1
    
    if models.get('transformers') and len(models['transformers']) > 0:
        print("🤖 Transformer Models:")
        for model_name, predictor in models['transformers'].items():
            print(f"   {model_name}:")
            print(f"      Model type: {predictor.model.config.model_type if predictor.model else 'Unknown'}")
            if predictor.model:
                print(f"      Parameters: {sum(p.numel() for p in predictor.model.parameters()):,}")
                print(f"      Max length: {predictor.model.config.max_position_embeddings if hasattr(predictor.model.config, 'max_position_embeddings') else 'Unknown'}")
            print(f"      Device: {predictor.device}")
            print(f"      Classes: {predictor.label_names}")
            print(f"      Model path: {predictor.model_path}")
            print()
        model_count += len(models['transformers'])
    
    if model_count > 0:
        print(f"✅ {model_count} model(s) ready for sentiment analysis!")
        
        # Show model comparison table
        if model_count >= 2:
            print(f"\n📊 Model Comparison Table:")
            print(f"{'Model':<25} {'Type':<15} {'Parameters':<15} {'Device':<10}")
            print("-" * 70)
            
            if models['tfidf_model'] is not None:
                print(f"{'TF-IDF + Logistic Regression':<25} {'Traditional':<15} {'N/A':<15} {'CPU':<10}")
            
            if models['bilstm_model'] is not None:
                params = sum(p.numel() for p in models['bilstm_model'].parameters())
                device = str(models['device']).split(':')[0].upper()
                print(f"{'BiLSTM + GloVe':<25} {'Neural':<15} {f'{params:,}':<15} {device:<10}")
            
            if models.get('transformers'):
                for model_name, predictor in models['transformers'].items():
                    if predictor.model:
                        params = sum(p.numel() for p in predictor.model.parameters())
                        device = str(predictor.device).split(':')[0].upper()
                        print(f"{model_name:<25} {'Transformer':<15} {f'{params:,}':<15} {device:<10}")
    else:
        print("❌ No models available for sentiment analysis!")

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
    
    if (models['tfidf_model'] is None and 
        models['bilstm_model'] is None and 
        len(models.get('transformers', {})) == 0):
        print("\n❌ No models available. Please train the models first:")
        print("   python tf-idf+logreg.py")
        print("   python bilstm_glove.py")
        print("   python transformer_finetuning.py")
        return
    
    # Print model summary
    print_model_summary(models)
    
    # Test sample texts
    print("\n" + "=" * 50)
    test_sample_texts(models)
    
    # Test batch texts
    print("\n" + "=" * 50)
    batch_results = test_batch_texts(models)
    
    # Show batch results summary
    if len(batch_results) > 0:
        print(f"\n📊 Batch Test Summary:")
        print(f"   Total texts tested: {len(batch_results)}")
        
        # Count available models
        available_models = []
        if 'tfidf_sentiment' in batch_results.columns:
            available_models.append('TF-IDF')
        if 'bilstm_sentiment' in batch_results.columns:
            available_models.append('BiLSTM')
        
        # Add transformer models
        for col in batch_results.columns:
            if col.endswith('_sentiment') and col not in ['tfidf_sentiment', 'bilstm_sentiment']:
                model_name = col.replace('_sentiment', '').upper()
                available_models.append(model_name)
        
        print(f"   Models available: {', '.join(available_models)}")
        
        # Visualize results
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
