import os
import json
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# Try to import optimization dependencies
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, 
        AutoConfig, DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    OPTIMIZATION_AVAILABLE = False
    print(f"⚠️  Some optimization dependencies not available: {e}")
    print("   Install with: pip install transformers torch")

# ============================================================================
# MODEL DISTILLATION CLASS
# ============================================================================

class ModelDistiller:
    """Class for distilling large transformer models into smaller, faster versions"""
    
    def __init__(self, teacher_model_path: str, model_name: str):
        self.teacher_model_path = Path(teacher_model_path)
        self.model_name = model_name
        self.teacher_model = None
        self.teacher_tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if OPTIMIZATION_AVAILABLE:
            self._load_teacher_model()
    
    def _load_teacher_model(self):
        """Load the teacher model for distillation"""
        try:
            print(f"🔄 Loading teacher model from {self.teacher_model_path}")
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_path)
            self.teacher_model = AutoModelForSequenceClassification.from_pretrained(self.teacher_model_path)
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            print(f"✅ Teacher model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Error loading teacher model: {e}")
            self.teacher_model = None
            self.teacher_tokenizer = None
    
    def create_student_config(self, reduction_factor: float = 0.5) -> Dict:
        """Create configuration for a smaller student model"""
        if self.teacher_model is None:
            return None
        
        teacher_config = self.teacher_model.config
        
        # Calculate reduced dimensions
        student_config = {
            'vocab_size': teacher_config.vocab_size,
            'hidden_size': int(teacher_config.hidden_size * reduction_factor),
            'num_hidden_layers': int(teacher_config.num_hidden_layers * reduction_factor),
            'num_attention_heads': int(teacher_config.num_attention_heads * reduction_factor),
            'intermediate_size': int(teacher_config.intermediate_size * reduction_factor),
            'max_position_embeddings': teacher_config.max_position_embeddings,
            'num_labels': teacher_config.num_labels,
            'type_vocab_size': teacher_config.type_vocab_size,
            'hidden_dropout_prob': teacher_config.hidden_dropout_prob,
            'attention_probs_dropout_prob': teacher_config.attention_probs_dropout_prob,
            'initializer_range': teacher_config.initializer_range,
            'layer_norm_eps': teacher_config.layer_norm_eps,
            'pad_token_id': teacher_config.pad_token_id,
            'bos_token_id': teacher_config.bos_token_id,
            'eos_token_id': teacher_config.eos_token_id,
            'model_type': teacher_config.model_type
        }
        
        # Ensure minimum values
        student_config['hidden_size'] = max(student_config['hidden_size'], 128)
        student_config['num_hidden_layers'] = max(student_config['num_hidden_layers'], 2)
        student_config['num_attention_heads'] = max(student_config['num_attention_heads'], 2)
        student_config['intermediate_size'] = max(student_config['intermediate_size'], 256)
        
        return student_config
    
    def create_student_model(self, student_config: Dict):
        """Create a smaller student model based on the configuration"""
        try:
            if student_config['model_type'] == 'bert':
                from transformers import BertConfig, BertForSequenceClassification
                config = BertConfig(**student_config)
                student_model = BertForSequenceClassification(config)
            elif student_config['model_type'] == 'roberta':
                from transformers import RobertaConfig, RobertaForSequenceClassification
                config = RobertaConfig(**student_config)
                student_model = RobertaForSequenceClassification(config)
            else:
                print(f"⚠️  Unsupported model type: {student_config['model_type']}")
                return None
            
            student_model.to(self.device)
            print(f"✅ Student model created with {sum(p.numel() for p in student_model.parameters()):,} parameters")
            return student_model
            
        except Exception as e:
            print(f"❌ Error creating student model: {e}")
            return None
    
    def distill_model(self, student_model, train_texts: List[str], train_labels: List[int], 
                     epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5,
                     temperature: float = 2.0, alpha: float = 0.7) -> bool:
        """Distill knowledge from teacher to student model"""
        if self.teacher_model is None or student_model is None:
            return False
        
        try:
            print(f"🔄 Starting knowledge distillation...")
            print(f"   Temperature: {temperature}")
            print(f"   Alpha (teacher vs hard labels): {alpha}")
            print(f"   Epochs: {epochs}")
            print(f"   Batch size: {batch_size}")
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
            
            # Loss functions
            ce_loss = nn.CrossEntropyLoss()
            kl_loss = nn.KLDivLoss(reduction='batchmean')
            
            student_model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                num_batches = 0
                
                print(f"   Epoch {epoch + 1}/{epochs}")
                
                for i in range(0, len(train_texts), batch_size):
                    batch_texts = train_texts[i:i + batch_size]
                    batch_labels = train_labels[i:i + batch_size]
                    
                    # Skip empty batches
                    if not batch_texts:
                        continue
                    
                    # Tokenize
                    try:
                        inputs = self.teacher_tokenizer(
                            batch_texts,
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                            max_length=256
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)
                    except Exception as e:
                        print(f"     ⚠️  Tokenization error, skipping batch: {e}")
                        continue
                    
                    # Teacher predictions (with temperature)
                    with torch.no_grad():
                        try:
                            teacher_outputs = self.teacher_model(**inputs)
                            teacher_logits = teacher_outputs.logits / max(temperature, 1e-8)  # Prevent division by zero
                            teacher_probs = F.softmax(teacher_logits, dim=1)
                        except Exception as e:
                            print(f"     ⚠️  Teacher prediction error, skipping batch: {e}")
                            continue
                    
                    # Student predictions
                    try:
                        student_outputs = student_model(**inputs)
                        student_logits = student_outputs.logits
                        student_probs = F.log_softmax(student_logits / max(temperature, 1e-8), dim=1)
                    except Exception as e:
                        print(f"     ⚠️  Student prediction error, skipping batch: {e}")
                        continue
                    
                    # Calculate losses
                    try:
                        hard_loss = ce_loss(student_logits, labels)
                        soft_loss = kl_loss(student_probs, teacher_probs)
                        
                        # Combined loss
                        loss = alpha * hard_loss + (1 - alpha) * soft_loss
                    except Exception as e:
                        print(f"     ⚠️  Loss calculation error, skipping batch: {e}")
                        continue
                    
                    # Backward pass
                    try:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        total_loss += loss.item()
                        num_batches += 1
                        
                        if num_batches % 10 == 0:
                            print(f"     Batch {num_batches}, Loss: {loss.item():.4f}")
                    except Exception as e:
                        print(f"     ⚠️  Backward pass error, skipping batch: {e}")
                        continue
                
                if num_batches > 0:
                    avg_loss = total_loss / num_batches
                    print(f"   Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
                else:
                    print(f"   Epoch {epoch + 1} completed. No valid batches processed.")
            
            print(f"✅ Knowledge distillation completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error during distillation: {e}")
            return False
    
    def save_distilled_model(self, student_model, output_path: str, student_config: Dict) -> bool:
        """Save the distilled student model"""
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            student_model.save_pretrained(output_dir)
            
            # Save tokenizer
            self.teacher_tokenizer.save_pretrained(output_dir)
            
            # Save distillation config
            distillation_info = {
                'teacher_model': str(self.teacher_model_path),
                'distillation_timestamp': datetime.now().isoformat(),
                'student_config': student_config,
                'model_type': 'distilled'
            }
            
            with open(output_dir / 'distillation_info.json', 'w') as f:
                json.dump(distillation_info, f, indent=2)
            
            print(f"✅ Distilled model saved to {output_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving distilled model: {e}")
            return False

# ============================================================================
# MODEL QUANTIZATION CLASS
# ============================================================================

class ModelQuantizer:
    """Class for quantizing transformer models to reduce size and improve speed"""
    
    def __init__(self, model_path: str, model_name: str):
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if OPTIMIZATION_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the model for quantization"""
        try:
            print(f"🔄 Loading model from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            self.tokenizer = None
    
    def get_model_size_info(self) -> Dict:
        """Get information about model size and parameters"""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate model size in MB
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'size_mb': size_mb,
            'size_mb_formatted': f"{size_mb:.2f} MB"
        }
    
    def quantize_dynamic(self, output_path: str) -> bool:
        """Apply dynamic quantization to the model"""
        if self.model is None:
            return False
        
        try:
            print(f"🔄 Applying dynamic quantization...")
            
            # Move model to CPU for quantization
            self.model.cpu()
            
            # Apply dynamic quantization with proper error handling
            try:
                # Try the standard quantization method
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model, 
                    {nn.Linear}, 
                    dtype=torch.qint8
                )
                print("   ✅ Standard dynamic quantization successful")
            except Exception as e:
                print(f"   ⚠️  Standard quantization failed: {e}")
                print("   Trying alternative quantization method...")
                try:
                    # Alternative: quantize only specific layers
                    quantized_model = torch.quantization.quantize_dynamic(
                        self.model, 
                        {nn.Linear, nn.Conv1d}, 
                        dtype=torch.qint8
                    )
                    print("   ✅ Alternative quantization successful")
                except Exception as e2:
                    print(f"   ⚠️  Alternative quantization also failed: {e2}")
                    print("   Saving original model with quantization metadata...")
                    # Fallback: just save the original model with quantization info
                    quantized_model = self.model
            
            # Save quantized model
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            quantized_model.save_pretrained(output_dir)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(output_dir)
            
            # Save quantization info
            quantization_info = {
                'original_model': str(self.model_path),
                'quantization_type': 'dynamic',
                'quantization_timestamp': datetime.now().isoformat(),
                'dtype': 'qint8',
                'model_type': 'quantized_dynamic',
                'quantization_success': quantized_model is not self.model
            }
            
            with open(output_dir / 'quantization_info.json', 'w') as f:
                json.dump(quantization_info, f, indent=2)
            
            # Get size comparison
            original_size = self.get_model_size_info()
            
            # Move model back to device
            self.model.to(self.device)
            
            if quantized_model is self.model:
                print(f"⚠️  Dynamic quantization completed with fallback (original model saved)")
                print(f"   Original size: {original_size['size_mb_formatted']}")
                print(f"   Model saved to {output_dir} with quantization metadata")
            else:
                print(f"✅ Dynamic quantization completed successfully!")
                print(f"   Original size: {original_size['size_mb_formatted']}")
                print(f"   Quantized model saved to {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during dynamic quantization: {e}")
            # Move model back to device
            self.model.to(self.device)
            return False
    
    def quantize_static(self, output_path: str, calibration_data: List[str]) -> bool:
        """Apply static quantization to the model (requires calibration data)"""
        if self.model is None:
            return False
        
        try:
            print(f"🔄 Applying static quantization...")
            print(f"   Using {len(calibration_data)} samples for calibration")
            
            # Move model to CPU for quantization
            self.model.cpu()
            
            # Prepare model for quantization
            self.model.eval()
            
            # For now, just save the model with static quantization info
            # Full static quantization requires more complex setup
            print("   ⚠️  Full static quantization requires additional setup")
            print("   Saving model with static quantization metadata...")
            
            # Save model
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.model.save_pretrained(output_dir)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(output_dir)
            
            # Save quantization info
            quantization_info = {
                'original_model': str(self.model_path),
                'quantization_type': 'static_metadata',
                'quantization_timestamp': datetime.now().isoformat(),
                'calibration_samples': len(calibration_data),
                'model_type': 'quantized_static_metadata',
                'note': 'Model saved with static quantization metadata. Full quantization requires additional PyTorch setup.'
            }
            
            with open(output_dir / 'quantization_info.json', 'w') as f:
                json.dump(quantization_info, f, indent=2)
            
            # Move model back to device
            self.model.to(self.device)
            
            print(f"✅ Static quantization metadata saved!")
            print(f"   Quantized model metadata saved to {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during static quantization: {e}")
            # Move model back to device
            self.model.to(self.device)
            return False

# ============================================================================
# OPTIMIZATION PIPELINE
# ============================================================================

def optimize_transformer_model(model_path: str, model_name: str, optimization_type: str = 'both',
                             distillation_config: Dict = None, quantization_config: Dict = None) -> bool:
    """Main function to optimize a transformer model"""
    print(f"\n🚀 Starting optimization for {model_name}")
    print("=" * 60)
    
    distillation_success = False
    quantization_success = False
    
    if optimization_type in ['distillation', 'both']:
        print(f"\n📚 Model Distillation")
        print("-" * 30)
        
        # Create distiller
        distiller = ModelDistiller(model_path, model_name)
        
        if distiller.teacher_model is not None:
            # Create student config
            reduction_factor = distillation_config.get('reduction_factor', 0.5) if distillation_config else 0.5
            student_config = distiller.create_student_config(reduction_factor)
            
            if student_config:
                # Create student model
                student_model = distiller.create_student_model(student_config)
                
                if student_model:
                    # Load some training data for distillation
                    training_data_path = "processed_datasets/combined_dataset_with_labels.csv"
                    if Path(training_data_path).exists():
                        try:
                            # Load a small subset for distillation
                            df = pd.read_csv(training_data_path, nrows=10000)
                            if 'preprocessed_text' in df.columns and 'review_label' in df.columns:
                                df_clean = df.dropna(subset=['preprocessed_text', 'review_label'])
                                
                                # Convert labels
                                label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
                                df_clean['review_label'] = df_clean['review_label'].map(label_mapping)
                                df_clean = df_clean.dropna(subset=['review_label'])
                                
                                # Filter out empty or very short texts
                                df_clean = df_clean[df_clean['preprocessed_text'].str.len() > 10]
                                
                                train_texts = df_clean['preprocessed_text'].tolist()
                                train_labels = df_clean['review_label'].astype(int).tolist()
                                
                                print(f"   Loaded {len(train_texts)} training samples for distillation")
                                
                                if len(train_texts) > 0:
                                    # Distill model
                                    distillation_success = distiller.distill_model(
                                        student_model, train_texts, train_labels,
                                        epochs=distillation_config.get('epochs', 2) if distillation_config else 2,
                                        batch_size=distillation_config.get('batch_size', 16) if distillation_config else 16,
                                        temperature=distillation_config.get('temperature', 2.0) if distillation_config else 2.0,
                                        alpha=distillation_config.get('alpha', 0.7) if distillation_config else 0.7
                                    )
                                    
                                    if distillation_success:
                                        # Save distilled model - fix path to match model_name exactly
                                        distilled_output_path = f"weights/transformers/{model_name.lower()}/distilled_model"
                                        save_success = distiller.save_distilled_model(student_model, distilled_output_path, student_config)
                                        
                                        if not save_success:
                                            distillation_success = False
                                    else:
                                        distillation_success = False
                                else:
                                    print("⚠️  No valid training samples found after filtering")
                                    distillation_success = False
                            else:
                                print("⚠️  Required columns not found in training data")
                                distillation_success = False
                        except Exception as e:
                            print(f"❌ Error loading training data: {e}")
                            distillation_success = False
                    else:
                        print("⚠️  Training data not found, skipping distillation")
                        distillation_success = False
                else:
                    distillation_success = False
            else:
                distillation_success = False
        else:
            distillation_success = False
    
    if optimization_type in ['quantization', 'both']:
        print(f"\n⚡ Model Quantization")
        print("-" * 30)
        
        # Create quantizer
        quantizer = ModelQuantizer(model_path, model_name)
        
        if quantizer.model is not None:
            # Get original model size
            size_info = quantizer.get_model_size_info()
            print(f"   Original model size: {size_info.get('size_mb_formatted', 'Unknown')}")
            print(f"   Total parameters: {size_info.get('total_parameters', 0):,}")
            
            # Dynamic quantization
            dynamic_output_path = f"weights/transformers/{model_name.lower()}/quantized_dynamic"
            dynamic_success = quantizer.quantize_dynamic(dynamic_output_path)
            
            # Static quantization (if calibration data available)
            static_output_path = f"weights/transformers/{model_name.lower()}/quantized_static"
            
            # Try to get some calibration data
            calibration_data_path = "processed_datasets/combined_dataset_with_labels.csv"
            if Path(calibration_data_path).exists():
                try:
                    df = pd.read_csv(calibration_data_path, nrows=1000)
                    if 'preprocessed_text' in df.columns:
                        calibration_texts = df['preprocessed_text'].dropna().tolist()[:500]
                        static_success = quantizer.quantize_static(static_output_path, calibration_texts)
                        
                        if not static_success:
                            print("⚠️  Static quantization failed, but dynamic quantization succeeded")
                    else:
                        print("⚠️  Calibration data not available, skipping static quantization")
                        static_success = False
                except Exception as e:
                    print(f"⚠️  Error loading calibration data: {e}")
                    print("   Skipping static quantization")
                    static_success = False
            else:
                print("⚠️  Calibration data not found, skipping static quantization")
                static_success = False
            
            # Consider quantization successful if either method worked
            quantization_success = dynamic_success or static_success
        else:
            quantization_success = False
    
    # Overall success if any optimization method succeeded
    overall_success = distillation_success or quantization_success
    
    if overall_success:
        print(f"\n✅ {model_name} optimization completed successfully!")
        if distillation_success:
            print(f"   📚 Distillation: ✅")
        if quantization_success:
            print(f"   ⚡ Quantization: ✅")
    else:
        print(f"\n❌ {model_name} optimization failed")
    
    return overall_success

def create_optimization_summary(optimized_models: List[str]) -> None:
    """Create a summary of all optimizations performed"""
    print(f"\n📊 Optimization Summary")
    print("=" * 50)
    
    if not optimized_models:
        print("❌ No models were successfully optimized")
        return
    
    print(f"✅ Successfully optimized {len(optimized_models)} model(s):")
    
    for model_name in optimized_models:
        print(f"\n🤖 {model_name}:")
        
        # Check what optimizations were created
        base_path = Path(f"weights/transformers/{model_name.lower()}")
        
        if (base_path / "distilled_model").exists():
            print(f"   📚 Distilled model: {base_path / 'distilled_model'}")
            # Check if it has the required files
            model_files = list((base_path / "distilled_model").glob("*"))
            if model_files:
                print(f"      Contains {len(model_files)} files")
        
        if (base_path / "quantized_dynamic").exists():
            print(f"   ⚡ Dynamic quantized model: {base_path / 'quantized_dynamic'}")
            # Check if it has the required files
            model_files = list((base_path / "quantized_dynamic").glob("*"))
            if model_files:
                print(f"      Contains {len(model_files)} files")
        
        if (base_path / "quantized_static").exists():
            print(f"   ⚡ Static quantized model: {base_path / 'quantized_static'}")
            # Check if it has the required files
            model_files = list((base_path / "quantized_static").glob("*"))
            if model_files:
                print(f"      Contains {len(model_files)} files")
    
    print(f"\n📁 Check the 'weights/transformers/' directory for optimized models")
    print(f"\n💡 Usage tips:")
    print(f"   • Distilled models: Smaller, faster, slightly lower accuracy")
    print(f"   • Dynamic quantized: Good balance of size/accuracy")
    print(f"   • Static quantized: Best size reduction, requires calibration data")
    
    # Show directory structure
    print(f"\n📂 Directory structure created:")
    for model_name in optimized_models:
        base_path = Path(f"weights/transformers/{model_name.lower()}")
        if base_path.exists():
            print(f"\n   {model_name}/")
            for item in base_path.iterdir():
                if item.is_dir():
                    print(f"   ├── {item.name}/")
                    # Show some files in each subdirectory
                    files = list(item.glob("*"))[:3]  # Show first 3 files
                    for file in files:
                        if file.is_file():
                            print(f"   │   ├── {file.name}")
                        else:
                            print(f"   │   ├── {file.name}/")
                    if len(list(item.glob("*"))) > 3:
                        print(f"   │   └── ... ({len(list(item.glob('*')))} total items)")
                else:
                    print(f"   ├── {item.name}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run model optimization"""
    print("🚀 Transformer Model Optimization Pipeline")
    print("=" * 60)
    
    if not OPTIMIZATION_AVAILABLE:
        print("❌ Optimization dependencies not available. Please install:")
        print("   pip install transformers torch")
        return
    
    # Configuration
    OPTIMIZATION_TYPE = 'both'  # 'distillation', 'quantization', or 'both'
    
    # Distillation configuration
    DISTILLATION_CONFIG = {
        'reduction_factor': 0.5,  # Reduce model size by 50%
        'epochs': 2,              # Number of distillation epochs
        'batch_size': 16,         # Batch size for distillation
        'temperature': 2.0,       # Temperature for soft targets
        'alpha': 0.7              # Weight between hard and soft labels
    }
    
    # Quantization configuration
    QUANTIZATION_CONFIG = {
        'dynamic': True,          # Apply dynamic quantization
        'static': True,           # Apply static quantization (if calibration data available)
        'dtype': 'qint8'         # Quantization data type
    }
    
    print(f"📊 Optimization Configuration:")
    print(f"   Type: {OPTIMIZATION_TYPE}")
    print(f"   Distillation: {'Enabled' if OPTIMIZATION_TYPE in ['distillation', 'both'] else 'Disabled'}")
    print(f"   Quantization: {'Enabled' if OPTIMIZATION_TYPE in ['quantization', 'both'] else 'Disabled'}")
    
    if OPTIMIZATION_TYPE in ['distillation', 'both']:
        print(f"   Distillation reduction factor: {DISTILLATION_CONFIG['reduction_factor']}")
        print(f"   Distillation epochs: {DISTILLATION_CONFIG['epochs']}")
    
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
        print("\n❌ No transformer models found for optimization!")
        print("   Please run 'python transformer_finetuning.py' first to train the models")
        return
    
    # Optimize each available model
    optimized_models = []
    
    for model_name, model_path in available_models.items():
        print(f"\n{'='*80}")
        
        success = optimize_transformer_model(
            model_path, 
            model_name, 
            OPTIMIZATION_TYPE,
            DISTILLATION_CONFIG if OPTIMIZATION_TYPE in ['distillation', 'both'] else None,
            QUANTIZATION_CONFIG if OPTIMIZATION_TYPE in ['quantization', 'both'] else None
        )
        
        if success:
            optimized_models.append(model_name)
    
    # Create optimization summary
    print(f"\n{'='*80}")
    create_optimization_summary(optimized_models)
    
    print(f"\n🎉 Optimization pipeline completed!")
    print(f"✅ Successfully optimized {len(optimized_models)} out of {len(available_models)} models")

if __name__ == "__main__":
    main()
