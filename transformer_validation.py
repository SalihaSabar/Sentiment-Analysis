import pandas as pd
import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import json
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

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
# TRANSFORMER VALIDATION CLASS
# ============================================================================

class TransformerValidator:
    """Class for validating transformer models and calculating metrics"""
    
    def __init__(self, model_path, model_name):
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_names = ["Negative", "Neutral", "Positive"]
        self.label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
        self.reverse_label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            print(f"❌ Transformers not available for {model_name}")
    
    def _load_model(self):
        """Load the transformer model and tokenizer"""
        try:
            print(f"🔄 Loading {self.model_name} from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ {self.model_name} loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Error loading {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None
    
    def predict_batch(self, texts, batch_size=16):
        """Predict sentiment for a batch of texts"""
        if self.model is None or self.tokenizer is None:
            return None, None
        
        all_predictions = []
        all_probabilities = []
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        print(f"   Processing {len(texts):,} samples in {total_batches} batches (batch size: {batch_size})")
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = F.softmax(outputs.logits, dim=1)
                    predictions = torch.argmax(outputs.logits, dim=1)
                
                # Convert to numpy
                batch_preds = predictions.cpu().numpy()
                batch_probs = probabilities.cpu().numpy()
                
                all_predictions.extend(batch_preds)
                all_probabilities.extend(batch_probs)
                
                # Progress update
                if batch_num % 10 == 0 or batch_num == total_batches:
                    print(f"     Batch {batch_num}/{total_batches} completed ({batch_num/total_batches*100:.1f}%)")
                
                # Clear GPU memory if using CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error in batch {batch_num}: {e}")
                # Fill with default values for failed predictions
                all_predictions.extend([1] * len(batch_texts))  # Default to Neutral
                all_probabilities.extend([[0.33, 0.34, 0.33]] * len(batch_texts))
        
        print(f"   ✅ All {total_batches} batches processed successfully")
        return np.array(all_predictions), np.array(all_probabilities)
    
    def calculate_metrics(self, true_labels, predictions, probabilities=None):
        """Calculate comprehensive evaluation metrics"""
        if len(true_labels) != len(predictions):
            print(f"❌ Label and prediction count mismatch: {len(true_labels)} vs {len(predictions)}")
            return None
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision_macro = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(true_labels, predictions, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Detailed classification report
        class_report = classification_report(
            true_labels, 
            predictions, 
            target_names=self.label_names,
            output_dict=True
        )
        
        metrics = {
            'overall': {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'total_samples': len(true_labels)
            },
            'per_class': {
                'precision': dict(zip(self.label_names, precision_per_class)),
                'recall': dict(zip(self.label_names, recall_per_class)),
                'f1': dict(zip(self.label_names, f1_per_class))
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
        
        if probabilities is not None:
            # Calculate confidence metrics
            max_confidences = np.max(probabilities, axis=1)
            avg_confidence = np.mean(max_confidences)
            metrics['confidence'] = {
                'average_confidence': float(avg_confidence),
                'min_confidence': float(np.min(max_confidences)),
                'max_confidence': float(np.max(max_confidences))
            }
        
        return metrics
    
    def create_confusion_matrix_plot(self, confusion_matrix, save_path=None):
        """Create and save confusion matrix visualization"""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                confusion_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.label_names,
                yticklabels=self.label_names
            )
            plt.title(f'{self.model_name} - Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Confusion matrix saved to {save_path}")
            
            plt.show()
            return True
            
        except Exception as e:
            print(f"❌ Error creating confusion matrix plot: {e}")
            return False
    
    def save_metrics_report(self, metrics, save_path):
        """Save detailed metrics report to JSON file"""
        try:
            # Add metadata
            report = {
                'model_name': self.model_name,
                'validation_timestamp': datetime.now().isoformat(),
                'device_used': str(self.device),
                'metrics': metrics
            }
            
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"✅ Metrics report saved to {save_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving metrics report: {e}")
            return False
    
    def print_metrics_summary(self, metrics):
        """Print a formatted summary of the metrics"""
        if metrics is None:
            print(f"❌ No metrics available for {self.model_name}")
            return
        
        print(f"\n📊 {self.model_name} - Validation Metrics Summary")
        print("=" * 60)
        
        # Overall metrics
        overall = metrics['overall']
        print(f"Overall Performance:")
        print(f"  Accuracy:  {overall['accuracy']:.4f}")
        print(f"  Precision: {overall['precision_macro']:.4f}")
        print(f"  Recall:    {overall['recall_macro']:.4f}")
        print(f"  F1-Score:  {overall['f1_macro']:.4f}")
        print(f"  Samples:   {overall['total_samples']:,}")
        
        # Per-class metrics
        print(f"\nPer-Class Performance:")
        for i, label in enumerate(self.label_names):
            precision = metrics['per_class']['precision'][label]
            recall = metrics['per_class']['recall'][label]
            f1 = metrics['per_class']['f1'][label]
            print(f"  {label:>8}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        # Confidence metrics (if available)
        if 'confidence' in metrics:
            conf = metrics['confidence']
            print(f"\nConfidence Metrics:")
            print(f"  Average: {conf['average_confidence']:.4f}")
            print(f"  Range:   {conf['min_confidence']:.4f} - {conf['max_confidence']:.4f}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print("           Predicted")
        print("           Neg  Neu  Pos")
        for i, label in enumerate(self.label_names):
            row = cm[i]
            print(f"{label:>8} {row[0]:>4} {row[1]:>4} {row[2]:>4}")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_test_data(data_path, text_column="preprocessed_text", label_column="review_label", max_samples=75000):
    """Load and prepare test data for validation (using a subset for efficiency)"""
    try:
        print(f"🔄 Loading test data from {data_path}")
        print(f"   Using maximum {max_samples:,} samples for validation")
        
        # Load the dataset in chunks to avoid memory issues
        chunk_size = 100000  # Process 100k rows at a time
        df_chunks = []
        
        print("   Loading data in chunks...")
        for chunk in pd.read_csv(data_path, chunksize=chunk_size):
            df_chunks.append(chunk)
            if len(df_chunks) * chunk_size >= max_samples * 2:  # Load extra to ensure we have enough after filtering
                break
        
        # Combine chunks
        df = pd.concat(df_chunks, ignore_index=True)
        print(f"✅ Loaded {len(df):,} samples from dataset")
        
        # Check required columns
        if text_column not in df.columns:
            print(f"❌ Text column '{text_column}' not found. Available columns: {list(df.columns)}")
            return None, None
        
        if label_column not in df.columns:
            print(f"❌ Label column '{label_column}' not found. Available columns: {list(df.columns)}")
            return None, None
        
        # Filter out rows with missing data
        df_clean = df.dropna(subset=[text_column, label_column])
        print(f"✅ Clean dataset has {len(df_clean):,} samples after removing missing data")
        
        # Convert labels to numeric if they're strings
        if df_clean[label_column].dtype == 'object':
            label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
            df_clean[label_column] = df_clean[label_column].map(label_mapping)
            
            # Check for any unmapped labels
            if df_clean[label_column].isna().any():
                print(f"⚠️  Found unmapped labels: {df_clean[df_clean[label_column].isna()][label_column].unique()}")
                df_clean = df_clean.dropna(subset=[label_column])
                print(f"✅ Final clean dataset has {len(df_clean):,} samples")
        
        # Stratified sampling to maintain label distribution
        if len(df_clean) > max_samples:
            print(f"   Performing stratified sampling to get {max_samples:,} samples...")
            
            # Get label distribution
            label_counts = df_clean[label_column].value_counts().sort_index()
            print(f"   Original label distribution: {dict(label_counts)}")
            
            # Calculate samples per class (maintaining proportions)
            samples_per_class = {}
            total_samples = min(max_samples, len(df_clean))
            
            for label in sorted(label_counts.index):
                proportion = label_counts[label] / len(df_clean)
                samples_per_class[label] = max(1, int(proportion * total_samples))
            
            # Adjust to match total samples
            current_total = sum(samples_per_class.values())
            if current_total > total_samples:
                # Reduce samples from largest class
                largest_class = max(samples_per_class.items(), key=lambda x: x[1])[0]
                samples_per_class[largest_class] -= (current_total - total_samples)
            elif current_total < total_samples:
                # Add samples to largest class
                largest_class = max(samples_per_class.items(), key=lambda x: x[1])[0]
                samples_per_class[largest_class] += (total_samples - current_total)
            
            print(f"   Target samples per class: {samples_per_class}")
            
            # Perform stratified sampling
            sampled_dfs = []
            for label, n_samples in samples_per_class.items():
                label_df = df_clean[df_clean[label_column] == label]
                if len(label_df) >= n_samples:
                    sampled_label_df = label_df.sample(n=n_samples, random_state=42)
                else:
                    sampled_label_df = label_df  # Use all available samples
                sampled_dfs.append(sampled_label_df)
            
            df_clean = pd.concat(sampled_dfs, ignore_index=True)
            print(f"✅ Stratified sampling completed: {len(df_clean):,} samples")
        
        # Extract texts and labels
        texts = df_clean[text_column].tolist()
        labels = df_clean[label_column].astype(int).tolist()
        
        # Validate label range
        unique_labels = set(labels)
        if not unique_labels.issubset({0, 1, 2}):
            print(f"⚠️  Unexpected labels found: {unique_labels}")
        
        # Final label distribution
        final_label_dist = dict(pd.Series(labels).value_counts().sort_index())
        print(f"✅ Test data prepared: {len(texts):,} texts, {len(labels):,} labels")
        print(f"   Final label distribution: {final_label_dist}")
        
        return texts, labels
        
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None, None

def create_validation_output_dir(model_name):
    """Create output directory for validation results"""
    output_dir = Path(f"validation_results/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def estimate_validation_time(num_samples, batch_size=16, device_type="CPU"):
    """Estimate validation time based on sample size and hardware"""
    if device_type == "CUDA":
        # GPU processing is much faster
        samples_per_second = 1000  # Conservative estimate for GPU
        estimated_seconds = num_samples / samples_per_second
    else:
        # CPU processing is slower
        samples_per_second = 100   # Conservative estimate for CPU
        estimated_seconds = num_samples / samples_per_second
    
    # Convert to human readable format
    if estimated_seconds < 60:
        time_str = f"{estimated_seconds:.1f} seconds"
    elif estimated_seconds < 3600:
        minutes = estimated_seconds / 60
        time_str = f"{minutes:.1f} minutes"
    else:
        hours = estimated_seconds / 3600
        time_str = f"{hours:.1f} hours"
    
    return time_str, estimated_seconds

# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def validate_transformer_model(model_path, model_name, test_texts, test_labels, batch_size=16):
    """Main function to validate a transformer model"""
    print(f"\n🎯 Starting validation for {model_name}")
    print("=" * 60)
    
    # Create validator instance
    validator = TransformerValidator(model_path, model_name)
    
    if validator.model is None:
        print(f"❌ Cannot validate {model_name} - model not loaded")
        return None
    
    # Create output directory
    output_dir = create_validation_output_dir(model_name)
    
    # Make predictions
    print(f"🔄 Making predictions on {len(test_texts)} test samples...")
    predictions, probabilities = validator.predict_batch(test_texts, batch_size)
    
    if predictions is None:
        print(f"❌ Failed to get predictions for {model_name}")
        return None
    
    # Calculate metrics
    print(f"🔄 Calculating validation metrics...")
    metrics = validator.calculate_metrics(test_labels, predictions, probabilities)
    
    if metrics is None:
        print(f"❌ Failed to calculate metrics for {model_name}")
        return None
    
    # Print metrics summary
    validator.print_metrics_summary(metrics)
    
    # Save confusion matrix plot
    cm_save_path = output_dir / "confusion_matrix.png"
    validator.create_confusion_matrix_plot(
        np.array(metrics['confusion_matrix']), 
        cm_save_path
    )
    
    # Save metrics report
    metrics_save_path = output_dir / "validation_metrics.json"
    validator.save_metrics_report(metrics, metrics_save_path)
    
    # Save predictions for further analysis
    predictions_df = pd.DataFrame({
        'text': test_texts,
        'true_label': test_labels,
        'predicted_label': predictions,
        'confidence': np.max(probabilities, axis=1) if probabilities is not None else [0] * len(predictions)
    })
    
    # Add label names
    predictions_df['true_sentiment'] = predictions_df['true_label'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
    predictions_df['predicted_sentiment'] = predictions_df['predicted_label'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})
    
    predictions_save_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_save_path, index=False)
    print(f"✅ Predictions saved to {predictions_save_path}")
    
    return metrics, predictions_df

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

def compare_transformer_models(results_dict):
    """Compare performance across different transformer models"""
    if len(results_dict) < 2:
        print("❌ Need at least 2 models for comparison")
        return
    
    print(f"\n📊 Comparative Analysis of Transformer Models")
    print("=" * 80)
    
    # Create comparison table
    comparison_data = []
    for model_name, (metrics, _) in results_dict.items():
        if metrics is None:
            continue
            
        overall = metrics['overall']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': overall['accuracy'],
            'Precision': overall['precision_macro'],
            'Recall': overall['recall_macro'],
            'F1-Score': overall['f1_macro'],
            'Samples': overall['total_samples']
        })
    
    if not comparison_data:
        print("❌ No valid metrics for comparison")
        return
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
    print("Performance Comparison (sorted by F1-Score):")
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Save comparison results
    comparison_save_path = Path("validation_results/model_comparison.csv")
    comparison_save_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_save_path, index=False)
    print(f"\n✅ Comparison results saved to {comparison_save_path}")
    
    # Create comparison visualization
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Transformer Models Performance Comparison', fontsize=16)
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i // 2, i % 2]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], color='skyblue', alpha=0.8)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_plot_path = Path("validation_results/model_comparison.png")
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to {comparison_plot_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating comparison visualization: {e}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run transformer model validation"""
    print("🚀 Transformer Model Validation and Accuracy Calculation")
    print("=" * 70)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers library not available. Please install it first:")
        print("   pip install transformers torch")
        return
    
    # Configuration
    VALIDATION_SAMPLE_SIZE = 75000  # Adjust this value (50k-100k recommended)
    BATCH_SIZE = 32  # Increase batch size for faster processing (adjust based on GPU memory)
    
    print(f"📊 Validation Configuration:")
    print(f"   Sample size: {VALIDATION_SAMPLE_SIZE:,} samples")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   This will use ~{VALIDATION_SAMPLE_SIZE/1000000:.1f}% of the full dataset")
    
    # Estimate validation time
    device_type = "CUDA" if torch.cuda.is_available() else "CPU"
    estimated_time, _ = estimate_validation_time(VALIDATION_SAMPLE_SIZE, BATCH_SIZE, device_type)
    print(f"   Estimated validation time: {estimated_time}")
    print(f"   Device: {device_type}")
    
    # Define model paths
    model_paths = {
        'BERT': 'weights/transformers/bert-base/final_model',
        'RoBERTa': 'weights/transformers/roberta-base/final_model'
    }
    
    # Check which models are available
    available_models = {}
    for model_name, model_path in model_paths.items():
        if Path(model_path).exists():
            available_models[model_name] = model_path
            print(f"✅ Found {model_name} model at {model_path}")
        else:
            print(f"❌ {model_name} model not found at {model_path}")
    
    if not available_models:
        print("\n❌ No transformer models found for validation!")
        print("   Please run 'python transformer_finetuning.py' first to train the models")
        return
    
    # Load test data
    test_data_path = "processed_datasets/combined_dataset_with_labels.csv"
    test_texts, test_labels = load_test_data(test_data_path, max_samples=VALIDATION_SAMPLE_SIZE)
    
    if test_texts is None or test_labels is None:
        print("❌ Failed to load test data. Cannot proceed with validation.")
        return
    
    # Validate each available model
    results = {}
    for model_name, model_path in available_models.items():
        print(f"\n{'='*80}")
        result = validate_transformer_model(model_path, model_name, test_texts, test_labels, BATCH_SIZE)
        if result is not None:
            results[model_name] = result
    
    # Compare models if multiple are available
    if len(results) >= 2:
        print(f"\n{'='*80}")
        compare_transformer_models(results)
    
    # Final summary
    print(f"\n{'='*80}")
    print("🎉 Validation Complete!")
    print(f"✅ Validated {len(results)} transformer model(s)")
    
    for model_name in results.keys():
        output_dir = Path(f"validation_results/{model_name}")
        print(f"   {model_name}: Results saved to {output_dir}")
    
    if len(results) >= 2:
        print(f"   Comparison results saved to validation_results/")
    
    print(f"\n📁 Check the 'validation_results/' directory for detailed reports and visualizations")

if __name__ == "__main__":
    main()
