# # fair_physics_comparison.py
# # Fair comparison between Custom Physics ViT, DINO, and CLIP

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import torch
# from PIL import Image
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression, Ridge
# from sklearn.metrics import (
#     accuracy_score, f1_score, confusion_matrix,
#     mean_squared_error, mean_absolute_error, r2_score,
#     silhouette_score
# )
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler

# # Try to import transformers
# try:
#     from transformers import AutoImageProcessor, AutoModel
#     TRANSFORMERS_AVAILABLE = True
#     print("✓ Transformers available")
# except ImportError:
#     TRANSFORMERS_AVAILABLE = False
#     print("❌ Transformers not available. Install with: pip install transformers")

# class PhysicsEmbeddingExtractor:
#     """Extract embeddings from physics images using different models"""
    
#     def __init__(self, device='cpu'):
#         self.device = device
#         print(f"Using device: {device}")
        
#     def load_dino(self):
#         """Load DINO v2 model"""
#         print("Loading DINO v2...")
#         self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
#         self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
#         self.dino_model.eval()
#         print("✓ DINO v2 loaded")
        
#     def load_clip(self):
#         """Load CLIP model"""
#         print("Loading CLIP...")
#         self.clip_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-base-patch32')
#         self.clip_model = AutoModel.from_pretrained('openai/clip-vit-base-patch32').to(self.device)
#         self.clip_model.eval()
#         print("✓ CLIP loaded")
    
#     def extract_dino_embeddings(self, image_paths, batch_size=16):
#         """Extract DINO embeddings"""
#         print(f"Extracting DINO embeddings from {len(image_paths)} images...")
#         embeddings = []
        
#         with torch.no_grad():
#             for i in range(0, len(image_paths), batch_size):
#                 batch_paths = image_paths[i:i+batch_size]
                
#                 # Load images
#                 images = []
#                 for path in batch_paths:
#                     try:
#                         img = Image.open(path).convert('RGB')
#                         images.append(img)
#                     except Exception as e:
#                         print(f"Error loading {path}: {e}")
#                         images.append(Image.new('RGB', (224, 224), color=(128, 128, 128)))
                
#                 # Process batch
#                 inputs = self.dino_processor(images, return_tensors="pt").to(self.device)
#                 outputs = self.dino_model(**inputs)
                
#                 # Get CLS token embeddings
#                 batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
#                 embeddings.extend(batch_embeddings)
                
#                 if (i // batch_size + 1) % 10 == 0:
#                     print(f"  Processed {i + len(batch_paths)}/{len(image_paths)} images")
        
#         return np.array(embeddings)
    
#     def extract_clip_embeddings(self, image_paths, batch_size=16):
#         """Extract CLIP vision embeddings"""
#         print(f"Extracting CLIP embeddings from {len(image_paths)} images...")
#         embeddings = []
        
#         with torch.no_grad():
#             for i in range(0, len(image_paths), batch_size):
#                 batch_paths = image_paths[i:i+batch_size]
                
#                 # Load images
#                 images = []
#                 for path in batch_paths:
#                     try:
#                         img = Image.open(path).convert('RGB')
#                         images.append(img)
#                     except Exception as e:
#                         print(f"Error loading {path}: {e}")
#                         images.append(Image.new('RGB', (224, 224), color=(128, 128, 128)))
                
#                 # Process batch
#                 inputs = self.clip_processor(images, return_tensors="pt").to(self.device)
#                 outputs = self.clip_model.vision_model(**inputs)
                
#                 # Get pooled vision embeddings
#                 batch_embeddings = outputs.pooler_output.cpu().numpy()
#                 embeddings.extend(batch_embeddings)
                
#                 if (i // batch_size + 1) % 10 == 0:
#                     print(f"  Processed {i + len(batch_paths)}/{len(image_paths)} images")
        
#         return np.array(embeddings)

# class PhysicsEmbeddingComparator:
#     """Compare different embedding approaches for physics tasks"""
    
#     def __init__(self, custom_embeddings, labels, image_paths):
#         self.custom_embeddings = custom_embeddings
#         self.labels = labels
#         self.image_paths = image_paths
        
#         # Ensure all arrays have same length
#         min_len = min(len(custom_embeddings), len(labels), len(image_paths))
#         self.custom_embeddings = custom_embeddings[:min_len]
#         self.labels = labels[:min_len]
#         self.image_paths = image_paths[:min_len]
        
#         print(f"✓ Initialized with {min_len} samples")
#         print(f"✓ Custom embedding dim: {self.custom_embeddings.shape[1]}")
#         print(f"✓ Unique classes: {np.unique(self.labels)}")
        
#         # Storage for all embeddings
#         self.all_embeddings = {
#             'Custom Physics ViT': self.custom_embeddings
#         }
        
#     def extract_baseline_embeddings(self):
#         """Extract DINO and CLIP embeddings for same images"""
#         if not TRANSFORMERS_AVAILABLE:
#             print("❌ Skipping DINO/CLIP - transformers not available")
#             return
            
#         extractor = PhysicsEmbeddingExtractor(device='cpu')
        
#         # Extract DINO embeddings
#         try:
#             extractor.load_dino()
#             dino_embeddings = extractor.extract_dino_embeddings(self.image_paths)
#             self.all_embeddings['DINO v2'] = dino_embeddings
#             print(f"✓ DINO embeddings: {dino_embeddings.shape}")
#         except Exception as e:
#             print(f"❌ DINO extraction failed: {e}")
        
#         # Extract CLIP embeddings  
#         try:
#             extractor.load_clip()
#             clip_embeddings = extractor.extract_clip_embeddings(self.image_paths)
#             self.all_embeddings['CLIP'] = clip_embeddings
#             print(f"✓ CLIP embeddings: {clip_embeddings.shape}")
#         except Exception as e:
#             print(f"❌ CLIP extraction failed: {e}")
    
#     def run_classification_comparison(self):
#         """Compare classification performance across all embeddings"""
#         print("\n" + "="*60)
#         print("🎯 CLASSIFICATION COMPARISON")
#         print("="*60)
        
#         results = {}
        
#         for model_name, embeddings in self.all_embeddings.items():
#             print(f"\n--- {model_name} ---")
            
#             # Check if we have enough classes
#             unique_classes = np.unique(self.labels)
#             if len(unique_classes) < 2:
#                 print(f"❌ Only {len(unique_classes)} classes available")
#                 continue
                
#             # Normalize embeddings
#             scaler = StandardScaler()
#             embeddings_norm = scaler.fit_transform(embeddings)
            
#             # Split data
#             X_train, X_test, y_train, y_test = train_test_split(
#                 embeddings_norm, self.labels, test_size=0.3, random_state=42, 
#                 stratify=self.labels
#             )
            
#             # Train classifier
#             clf = LogisticRegression(max_iter=2000, random_state=42)
#             clf.fit(X_train, y_train)
#             y_pred = clf.predict(X_test)
            
#             # Compute metrics
#             acc = accuracy_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred, average='weighted')
            
#             results[model_name] = {'accuracy': acc, 'f1': f1}
            
#             print(f"Accuracy: {acc:.4f}")
#             print(f"F1 Score: {f1:.4f}")
            
#             # Create confusion matrix
#             cm = confusion_matrix(y_test, y_pred)
#             plt.figure(figsize=(8, 6))
#             sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
#             plt.title(f"Confusion Matrix - {model_name}")
#             plt.xlabel("Predicted")
#             plt.ylabel("True")
#             plt.tight_layout()
#             plt.savefig(f"plots/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
#             plt.close()
        
#         return results
    
#     def run_temporal_forecasting_comparison(self):
#         """Compare temporal forecasting using consistent methodology"""
#         print("\n" + "="*60)
#         print("⏰ TEMPORAL FORECASTING COMPARISON")
#         print("="*60)
        
#         results = {}
        
#         for model_name, embeddings in self.all_embeddings.items():
#             print(f"\n--- {model_name} ---")
            
#             # Normalize embeddings
#             scaler = StandardScaler()
#             embeddings_norm = scaler.fit_transform(embeddings)
            
#             # Create temporal targets (simple next-step prediction)
#             # Note: For your 200-sample test, we use the same np.roll approach
#             # In full model, this will use proper trajectory sequences
#             targets = np.roll(embeddings_norm, shift=-1, axis=0)
            
#             # Remove last sample (wraps around)
#             X = embeddings_norm[:-1]
#             y = targets[:-1]
            
#             # Split data
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.3, random_state=42
#             )
            
#             # Train forecasting model
#             model = Ridge(alpha=1.0, random_state=42)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
            
#             # Compute metrics
#             mse = mean_squared_error(y_test, y_pred)
#             mae = mean_absolute_error(y_test, y_pred)
#             r2 = r2_score(y_test, y_pred)
            
#             results[model_name] = {'mse': mse, 'mae': mae, 'r2': r2}
            
#             print(f"MSE: {mse:.6f}")
#             print(f"MAE: {mae:.6f}")
#             print(f"R²: {r2:.4f}")
        
#         return results
    
#     def create_tsne_comparison(self):
#         """Create t-SNE plots for all embeddings"""
#         print("\n" + "="*60)
#         print("📊 t-SNE VISUALIZATION COMPARISON")
#         print("="*60)
        
#         n_models = len(self.all_embeddings)
#         fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
#         if n_models == 1:
#             axes = [axes]
        
#         for idx, (model_name, embeddings) in enumerate(self.all_embeddings.items()):
#             print(f"Computing t-SNE for {model_name}...")
            
#             # Normalize embeddings
#             scaler = StandardScaler()
#             embeddings_norm = scaler.fit_transform(embeddings)
            
#             # Compute t-SNE
#             perplexity = min(30, len(embeddings) // 4)
#             tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
#             reduced = tsne.fit_transform(embeddings_norm)
            
#             # Plot
#             ax = axes[idx]
#             scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
#                                c=self.labels, cmap='tab10', 
#                                s=20, alpha=0.7, edgecolors='k', linewidth=0.5)
#             ax.set_title(f"{model_name}")
#             ax.set_xlabel("t-SNE 1")
#             ax.set_ylabel("t-SNE 2")
            
#             # Compute silhouette score
#             try:
#                 sil_score = silhouette_score(reduced, self.labels)
#                 ax.text(0.02, 0.98, f'Silhouette: {sil_score:.3f}', 
#                        transform=ax.transAxes, verticalalignment='top',
#                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#             except:
#                 pass
        
#         # Add colorbar
#         plt.colorbar(scatter, ax=axes, label='Physics Class')
#         plt.tight_layout()
#         plt.savefig("plots/tsne_comparison.png", dpi=150, bbox_inches='tight')
#         plt.close()
        
#         print("✓ Saved t-SNE comparison to plots/tsne_comparison.png")
    
#     def create_summary_comparison(self, classification_results, forecasting_results):
#         """Create summary comparison chart"""
#         print("\n" + "="*60)
#         print("📈 SUMMARY COMPARISON")
#         print("="*60)
        
#         # Create comparison table
#         models = list(classification_results.keys())
        
#         # Classification comparison
#         fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
#         # Accuracy comparison
#         accuracies = [classification_results[m]['accuracy'] for m in models]
#         axes[0, 0].bar(models, accuracies, color=['#2E86AB', '#A23B72', '#F18F01'])
#         axes[0, 0].set_title('Classification Accuracy')
#         axes[0, 0].set_ylabel('Accuracy')
#         axes[0, 0].tick_params(axis='x', rotation=45)
        
#         # F1 Score comparison
#         f1_scores = [classification_results[m]['f1'] for m in models]
#         axes[0, 1].bar(models, f1_scores, color=['#2E86AB', '#A23B72', '#F18F01'])
#         axes[0, 1].set_title('Classification F1 Score')
#         axes[0, 1].set_ylabel('F1 Score')
#         axes[0, 1].tick_params(axis='x', rotation=45)
        
#         # R² comparison (forecasting)
#         r2_scores = [forecasting_results[m]['r2'] for m in models]
#         axes[1, 0].bar(models, r2_scores, color=['#2E86AB', '#A23B72', '#F18F01'])
#         axes[1, 0].set_title('Temporal Forecasting R²')
#         axes[1, 0].set_ylabel('R² Score')
#         axes[1, 0].tick_params(axis='x', rotation=45)
        
#         # MSE comparison (forecasting) - lower is better
#         mse_scores = [forecasting_results[m]['mse'] for m in models]
#         axes[1, 1].bar(models, mse_scores, color=['#2E86AB', '#A23B72', '#F18F01'])
#         axes[1, 1].set_title('Temporal Forecasting MSE (lower is better)')
#         axes[1, 1].set_ylabel('MSE')
#         axes[1, 1].tick_params(axis='x', rotation=45)
        
#         plt.tight_layout()
#         plt.savefig("plots/performance_comparison.png", dpi=150, bbox_inches='tight')
#         plt.close()
        
#         # Print summary table
#         print("\nSUMMARY RESULTS:")
#         print("-" * 80)
#         print(f"{'Model':<20} {'Accuracy':<10} {'F1':<10} {'R²':<10} {'MSE':<10}")
#         print("-" * 80)
        
#         for model in models:
#             acc = classification_results[model]['accuracy']
#             f1 = classification_results[model]['f1']
#             r2 = forecasting_results[model]['r2']
#             mse = forecasting_results[model]['mse']
#             print(f"{model:<20} {acc:<10.4f} {f1:<10.4f} {r2:<10.4f} {mse:<10.6f}")
        
#         # Highlight best performer
#         best_acc_model = max(models, key=lambda m: classification_results[m]['accuracy'])
#         best_r2_model = max(models, key=lambda m: forecasting_results[m]['r2'])
        
#         print("-" * 80)
#         print(f"🏆 Best Classification: {best_acc_model}")
#         print(f"🏆 Best Forecasting: {best_r2_model}")
        
#         print("✓ Saved performance comparison to plots/performance_comparison.png")


# def load_custom_physics_data():
#     """Load your existing custom physics embeddings and metadata"""
#     try:
#         # Load embeddings and labels
#         custom_embeddings = np.load("Embeddings/physics_embeddings_layer_11.npy")
#         labels = np.load("Embeddings/physics_labels.npy")
        
#         # Load metadata to get image paths
#         metadata_path = "/cloud_psc_homedir/jessicae-cloud/Jessica/Images/multi_physics_vit/metadata.csv"
#         metadata = pd.read_csv(metadata_path)
        
#         # Get image paths (first N samples to match embeddings)
#         data_dir = "/cloud_psc_homedir/jessicae-cloud/Jessica/Images/multi_physics_vit"
#         image_paths = [os.path.join(data_dir, path) for path in metadata['image_path'].head(len(custom_embeddings))]
        
#         # Filter out missing images
#         valid_indices = []
#         valid_paths = []
        
#         for i, path in enumerate(image_paths):
#             if os.path.exists(path):
#                 valid_indices.append(i)
#                 valid_paths.append(path)
        
#         # Filter embeddings and labels to match valid images
#         custom_embeddings = custom_embeddings[valid_indices]
#         labels = labels[valid_indices]
        
#         print(f"✓ Loaded {len(custom_embeddings)} custom embeddings")
#         print(f"✓ Found {len(valid_paths)} valid images")
#         print(f"✓ Classes: {np.unique(labels)}")
        
#         return custom_embeddings, labels, valid_paths
        
#     except Exception as e:
#         print(f"❌ Error loading custom data: {e}")
#         return None, None, None


# def main():
#     """Run comprehensive physics embedding comparison"""
#     print("🔬 Physics Foundation Model vs Baselines Comparison")
#     print("="*60)
    
#     # Create output directory
#     os.makedirs("plots", exist_ok=True)
    
#     # Load your custom physics data
#     custom_embeddings, labels, image_paths = load_custom_physics_data()
    
#     if custom_embeddings is None:
#         print("❌ Cannot load custom physics data")
#         return
    
#     # Initialize comparator
#     comparator = PhysicsEmbeddingComparator(custom_embeddings, labels, image_paths)
    
#     # Extract baseline embeddings (DINO, CLIP)
#     print("\n1. Extracting baseline embeddings...")
#     comparator.extract_baseline_embeddings()
    
#     # Run comparisons
#     print("\n2. Running classification comparison...")
#     classification_results = comparator.run_classification_comparison()
    
#     print("\n3. Running temporal forecasting comparison...")
#     forecasting_results = comparator.run_temporal_forecasting_comparison()
    
#     print("\n4. Creating t-SNE visualizations...")
#     comparator.create_tsne_comparison()
    
#     print("\n5. Creating summary comparison...")
#     comparator.create_summary_comparison(classification_results, forecasting_results)
    
#     print("\n" + "="*60)
#     print("✅ COMPARISON COMPLETE!")
#     print("📁 Check plots/ directory for visualizations:")
#     print("   - tsne_comparison.png: Class separation visualization") 
#     print("   - performance_comparison.png: Metric comparison")
#     print("   - confusion_matrix_*.png: Individual confusion matrices")
#     print("="*60)


# if __name__ == "__main__":
#     main()

# fair_physics_comparison.py
# Fair comparison between Custom Physics ViT, DINO, and CLIP

# fair_physics_comparison.py
# Fair comparison between Custom Physics ViT, DINO, and CLIP

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from PIL import Image
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Try to import transformers
try:
    from transformers import AutoImageProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
    print("✓ Transformers available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ Transformers not available. Install with: pip install transformers")

class PhysicsEmbeddingExtractor:
    """Extract embeddings from physics images using different models"""
    
    def __init__(self, device='cpu'):
        self.device = device
        print(f"Using device: {device}")
        
    def load_dino(self):
        """Load DINO v2 model"""
        print("Loading DINO v2...")
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        self.dino_model.eval()
        print("✓ DINO v2 loaded")
        
    def load_clip(self):
        """Load CLIP model"""
        print("Loading CLIP...")
        self.clip_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model = AutoModel.from_pretrained('openai/clip-vit-base-patch32').to(self.device)
        self.clip_model.eval()
        print("✓ CLIP loaded")
    
    def extract_dino_embeddings(self, image_paths, batch_size=16):
        """Extract DINO embeddings"""
        print(f"Extracting DINO embeddings from {len(image_paths)} images...")
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                
                # Load images
                images = []
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert('RGB')
                        images.append(img)
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
                        images.append(Image.new('RGB', (224, 224), color=(128, 128, 128)))
                
                # Process batch
                inputs = self.dino_processor(images, return_tensors="pt").to(self.device)
                outputs = self.dino_model(**inputs)
                
                # Get CLS token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
                embeddings.extend(batch_embeddings)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"  Processed {i + len(batch_paths)}/{len(image_paths)} images")
        
        return np.array(embeddings)
    
    def extract_clip_embeddings(self, image_paths, batch_size=16):
        """Extract CLIP vision embeddings"""
        print(f"Extracting CLIP embeddings from {len(image_paths)} images...")
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i+batch_size]
                
                # Load images
                images = []
                for path in batch_paths:
                    try:
                        img = Image.open(path).convert('RGB')
                        images.append(img)
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
                        images.append(Image.new('RGB', (224, 224), color=(128, 128, 128)))
                
                # Process batch
                inputs = self.clip_processor(images, return_tensors="pt").to(self.device)
                outputs = self.clip_model.vision_model(**inputs)
                
                # Get pooled vision embeddings
                batch_embeddings = outputs.pooler_output.cpu().numpy()
                embeddings.extend(batch_embeddings)
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"  Processed {i + len(batch_paths)}/{len(image_paths)} images")
        
        return np.array(embeddings)

class PhysicsEmbeddingComparator:
    """Compare different embedding approaches for physics tasks"""
    
    def __init__(self, custom_embeddings, labels, image_paths, metadata=None):
        self.custom_embeddings = custom_embeddings
        self.labels = labels
        self.image_paths = image_paths
        self.metadata = metadata
        
        # Ensure all arrays have same length
        min_len = min(len(custom_embeddings), len(labels), len(image_paths))
        self.custom_embeddings = custom_embeddings[:min_len]
        self.labels = labels[:min_len]
        self.image_paths = image_paths[:min_len]
        
        if metadata is not None:
            self.metadata = metadata.head(min_len)
        
        print(f"✓ Initialized with {min_len} samples")
        print(f"✓ Custom embedding dim: {self.custom_embeddings.shape[1]}")
        print(f"✓ Unique classes: {np.unique(self.labels)}")
        
        # Detect model type based on number of classes
        num_classes = len(np.unique(self.labels))
        if num_classes <= 6:
            self.model_type = "small_model"  # 200-sample model
            print(f"✓ Detected small model with {num_classes} classes")
            # Class names for small model
            self.class_names = {0: 'pressure', 1: 'velocity_magnitude'}
        else:
            self.model_type = "full_model"   # Full temporal model
            print(f"✓ Detected full model with {num_classes} classes")
            # Class names for full model (12+ physics domains)
            self.class_names = {
                0: 'acoustic_scattering_inclusions',
                1: 'acoustic_scattering_discontinuous', 
                2: 'acoustic_scattering_maze',
                3: 'active_matter',
                4: 'euler_multi_quadrants_openBC',
                5: 'euler_multi_quadrants_periodicBC',
                6: 'gray_scott_reaction_diffusion',
                7: 'helmholtz_staircase',
                8: 'planetswe',
                9: 'rayleigh_benard',
                10: 'rayleigh_benard_uniform',
                11: 'shear_flow',
                12: 'turbulent_radiative_layer_2D',
                13: 'viscoelastic_instability'
            }
        
        # Create readable class labels for current dataset
        unique_labels = np.unique(self.labels)
        self.label_names = [self.class_names.get(label, f'Class_{label}') for label in unique_labels]
        print(f"✓ Class mapping: {dict(zip(unique_labels, self.label_names))}")
        
        # Storage for all embeddings
        self.all_embeddings = {
            'Custom Physics ViT': self.custom_embeddings
        }
        
    def extract_baseline_embeddings(self):
        """Extract DINO and CLIP embeddings for same images (with caching)"""
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Skipping DINO/CLIP - transformers not available")
            return
        
        # Check for cached embeddings first
        cache_dir = "baseline_embeddings"
        os.makedirs(cache_dir, exist_ok=True)
        
        dino_cache = f"{cache_dir}/dino_embeddings_{len(self.image_paths)}.npy"
        clip_cache = f"{cache_dir}/clip_embeddings_{len(self.image_paths)}.npy"
        
        # Load or extract DINO embeddings
        if os.path.exists(dino_cache):
            print("✓ Loading cached DINO embeddings...")
            dino_embeddings = np.load(dino_cache)
            self.all_embeddings['DINO v2'] = dino_embeddings
            print(f"✓ DINO embeddings: {dino_embeddings.shape}")
        else:
            try:
                print("🔄 Extracting DINO embeddings (will cache for future use)...")
                extractor = PhysicsEmbeddingExtractor(device='cpu')
                extractor.load_dino()
                dino_embeddings = extractor.extract_dino_embeddings(self.image_paths)
                np.save(dino_cache, dino_embeddings)
                self.all_embeddings['DINO v2'] = dino_embeddings
                print(f"✓ DINO embeddings: {dino_embeddings.shape} (cached)")
            except Exception as e:
                print(f"❌ DINO extraction failed: {e}")
        
        # Load or extract CLIP embeddings  
        if os.path.exists(clip_cache):
            print("✓ Loading cached CLIP embeddings...")
            clip_embeddings = np.load(clip_cache)
            self.all_embeddings['CLIP'] = clip_embeddings
            print(f"✓ CLIP embeddings: {clip_embeddings.shape}")
        else:
            try:
                print("🔄 Extracting CLIP embeddings (will cache for future use)...")
                if 'extractor' not in locals():
                    extractor = PhysicsEmbeddingExtractor(device='cpu')
                extractor.load_clip()
                clip_embeddings = extractor.extract_clip_embeddings(self.image_paths)
                np.save(clip_cache, clip_embeddings)
                self.all_embeddings['CLIP'] = clip_embeddings
                print(f"✓ CLIP embeddings: {clip_embeddings.shape} (cached)")
            except Exception as e:
                print(f"❌ CLIP extraction failed: {e}")
    
    def run_classification_comparison(self):
        """Compare classification performance across all embeddings"""
        print("\n" + "="*60)
        print("🎯 CLASSIFICATION COMPARISON")
        print("="*60)
        
        results = {}
        
        for model_name, embeddings in self.all_embeddings.items():
            print(f"\n--- {model_name} ---")
            
            # Check if we have enough classes
            unique_classes = np.unique(self.labels)
            if len(unique_classes) < 2:
                print(f"❌ Only {len(unique_classes)} classes available")
                continue
                
            # Normalize embeddings
            scaler = StandardScaler()
            embeddings_norm = scaler.fit_transform(embeddings)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings_norm, self.labels, test_size=0.3, random_state=42, 
                stratify=self.labels
            )
            
            # Train classifier
            clf = LogisticRegression(max_iter=2000, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[model_name] = {'accuracy': acc, 'f1': f1}
            
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
            plt.title(f"Confusion Matrix - {model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(f"plots/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
            plt.close()
        
        return results
    
    def run_temporal_forecasting_comparison(self, metadata=None):
        """Compare temporal forecasting using proper trajectory sequences"""
        print("\n" + "="*60)
        print("⏰ TEMPORAL FORECASTING COMPARISON")
        print("="*60)
        
        results = {}
        
        # Check if we have trajectory information
        has_trajectory_info = (metadata is not None and 
                             'trajectory_id' in metadata.columns and 
                             'timestep' in metadata.columns)
        
        if not has_trajectory_info:
            print("⚠️  No trajectory info - using simple np.roll approach")
        
        for model_name, embeddings in self.all_embeddings.items():
            print(f"\n--- {model_name} ---")
            
            # Normalize embeddings
            scaler = StandardScaler()
            embeddings_norm = scaler.fit_transform(embeddings)
            
            if has_trajectory_info:
                # Use proper temporal sequences from trajectories
                forecasting_pairs = self._create_trajectory_pairs(embeddings_norm, metadata)
                
                if len(forecasting_pairs) < 10:
                    print(f"❌ Only {len(forecasting_pairs)} temporal pairs - falling back to np.roll")
                    X, y = self._create_simple_temporal_pairs(embeddings_norm)
                else:
                    X = np.array([pair['input'] for pair in forecasting_pairs])
                    y = np.array([pair['target'] for pair in forecasting_pairs])
                    trajectories = [pair['trajectory'] for pair in forecasting_pairs]
                    
                    # Split by trajectory to prevent data leakage
                    unique_trajs = list(set(trajectories))
                    train_trajs = set(unique_trajs[:int(0.7 * len(unique_trajs))])
                    
                    train_mask = np.array([t in train_trajs for t in trajectories])
                    X_train, X_test = X[train_mask], X[~train_mask]
                    y_train, y_test = y[train_mask], y[~train_mask]
                    
                    print(f"Using {len(forecasting_pairs)} trajectory pairs from {len(unique_trajs)} trajectories")
            else:
                # Fallback to simple approach for 200-sample model
                X, y = self._create_simple_temporal_pairs(embeddings_norm)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
            
            # Train forecasting model
            model = Ridge(alpha=1.0, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Compute metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[model_name] = {'mse': mse, 'mae': mae, 'r2': r2}
            
            print(f"MSE: {mse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"R²: {r2:.4f}")
        
        return results
    
    def _create_trajectory_pairs(self, embeddings, metadata):
        """Create proper temporal pairs from trajectory sequences"""
        forecasting_pairs = []
        
        # Group by trajectory
        trajectories = metadata.groupby('trajectory_id')
        
        for traj_id, traj_data in trajectories:
            if len(traj_data) < 2:
                continue
                
            # Sort by timestep
            traj_data = traj_data.sort_values('timestep')
            traj_indices = traj_data.index.values
            
            # Create consecutive pairs
            for i in range(len(traj_indices) - 1):
                current_idx = traj_indices[i]
                next_idx = traj_indices[i + 1]
                
                forecasting_pairs.append({
                    'input': embeddings[current_idx],
                    'target': embeddings[next_idx],
                    'trajectory': traj_id
                })
        
        return forecasting_pairs
    
    def _create_simple_temporal_pairs(self, embeddings):
        """Create simple temporal pairs using np.roll (for 200-sample model)"""
        targets = np.roll(embeddings, shift=-1, axis=0)
        return embeddings[:-1], targets[:-1]
    
    def create_pca_analysis(self):
        """Create comprehensive PCA analysis for all embeddings"""
        print("\n" + "="*60)
        print("📊 PCA ANALYSIS")
        print("="*60)
        
        n_models = len(self.all_embeddings)
        
        # Create figure with subplots for different analyses
        fig = plt.figure(figsize=(18, 12))
        
        # Color palette for models
        model_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for idx, (model_name, embeddings) in enumerate(self.all_embeddings.items()):
            print(f"Computing PCA for {model_name}...")
            
            # Normalize embeddings
            scaler = StandardScaler()
            embeddings_norm = scaler.fit_transform(embeddings)
            
            # Full PCA
            pca_full = PCA()
            X_pca_full = pca_full.fit_transform(embeddings_norm)
            
            # === Explained Variance Plot === #
            ax1 = plt.subplot(2, n_models, idx + 1)
            plt.plot(np.cumsum(pca_full.explained_variance_ratio_), 
                    marker='o', color=model_colors[idx % len(model_colors)], linewidth=2)
            plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80%')
            plt.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90%')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title(f'Explained Variance - {model_name}')
            plt.grid(True, alpha=0.3)
            if idx == 0:
                plt.legend(fontsize=8)
            
            # Find components for 80% and 90% variance
            var_80 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.8) + 1
            var_90 = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.9) + 1
            print(f"  {model_name}: {var_80} components for 80%, {var_90} components for 90% variance")
            
            # === PCA Scatter Plot === #
            ax2 = plt.subplot(2, n_models, idx + 1 + n_models)
            
            # Create color map for classes
            unique_labels = np.unique(self.labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = self.labels == label
                class_name = self.class_names.get(label, f'Class_{label}')
                plt.scatter(X_pca_full[mask, 0], X_pca_full[mask, 1], 
                           c=[colors[i]], label=class_name, s=20, alpha=0.7, edgecolors='k', linewidth=0.3)
            
            plt.xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]:.1%} variance)')
            plt.title(f'PCA Projection - {model_name}')
            
            # Add legend outside plot for first subplot only
            if idx == 0:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, 
                          title='Physics Domains', title_fontsize=9)
        
        plt.tight_layout()
        plt.savefig("plots/pca_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Saved PCA analysis to plots/pca_analysis.png")
        
        # === Detailed PCA Pairplot for Best Model === #
        best_model = list(self.all_embeddings.keys())[0]  # Use first model (Custom ViT)
        embeddings = list(self.all_embeddings.values())[0]
        
        print(f"Creating detailed PCA pairplot for {best_model}...")
        
        # Normalize and compute PCA
        scaler = StandardScaler()
        embeddings_norm = scaler.fit_transform(embeddings)
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(embeddings_norm)
        
        # Create DataFrame for pairplot
        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(5)])
        pca_df['Class'] = [self.class_names.get(label, f'Class_{label}') for label in self.labels]
        
        # Create pairplot
        plt.figure(figsize=(12, 10))
        g = sns.pairplot(pca_df, vars=[f'PC{i+1}' for i in range(5)], hue='Class', 
                        corner=True, palette='tab10', plot_kws={'alpha': 0.7, 's': 15})
        g.fig.suptitle(f'PCA Pairplot - {best_model}', y=1.02, fontsize=14)
        
        # Move legend outside
        g._legend.set_bbox_to_anchor((1.05, 0.8))
        g._legend.set_title('Physics Domains')
        
        plt.savefig("plots/pca_pairplot.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Saved detailed PCA pairplot to plots/pca_pairplot.png")
    
    def create_tsne_comparison(self):
        """Create t-SNE plots for all embeddings with proper class labels"""
        print("\n" + "="*60)
        print("📊 t-SNE VISUALIZATION COMPARISON")
        print("="*60)
        
        n_models = len(self.all_embeddings)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        # Create consistent color mapping for all plots
        unique_labels = np.unique(self.labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        for idx, (model_name, embeddings) in enumerate(self.all_embeddings.items()):
            print(f"Computing t-SNE for {model_name}...")
            
            # Normalize embeddings
            scaler = StandardScaler()
            embeddings_norm = scaler.fit_transform(embeddings)
            
            # Compute t-SNE
            perplexity = min(30, len(embeddings) // 4)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                       init='pca', n_iter=1000)
            reduced = tsne.fit_transform(embeddings_norm)
            
            # Plot with proper class names
            ax = axes[idx]
            
            # Plot each class separately to get proper legend
            for label in unique_labels:
                mask = self.labels == label
                class_name = self.class_names.get(label, f'Class_{label}')
                ax.scatter(reduced[mask, 0], reduced[mask, 1], 
                          c=[label_to_color[label]], label=class_name,
                          s=25, alpha=0.7, edgecolors='k', linewidth=0.3)
            
            ax.set_title(f"{model_name}", fontsize=12, fontweight='bold')
            ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
            ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Compute and display silhouette score
            try:
                sil_score = silhouette_score(reduced, self.labels)
                ax.text(0.02, 0.98, f'Silhouette: {sil_score:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       fontsize=9, fontweight='bold')
            except Exception as e:
                print(f"Could not compute silhouette score for {model_name}: {e}")
        
        # Create a single legend outside all plots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), 
                  title='Physics Domains', title_fontsize=12, fontsize=10)
        
        # Adjust layout to make room for legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        plt.savefig("plots/tsne_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Saved t-SNE comparison to plots/tsne_comparison.png")
    
    def create_summary_comparison(self, classification_results, forecasting_results):
        """Create summary comparison chart"""
        print("\n" + "="*60)
        print("📈 SUMMARY COMPARISON")
        print("="*60)
        
        # Create comparison table
        models = list(classification_results.keys())
        
        # Classification comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Accuracy comparison
        accuracies = [classification_results[m]['accuracy'] for m in models]
        axes[0, 0].bar(models, accuracies, color=['#2E86AB', '#A23B72', '#F18F01'])
        axes[0, 0].set_title('Classification Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        f1_scores = [classification_results[m]['f1'] for m in models]
        axes[0, 1].bar(models, f1_scores, color=['#2E86AB', '#A23B72', '#F18F01'])
        axes[0, 1].set_title('Classification F1 Score')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² comparison (forecasting)
        r2_scores = [forecasting_results[m]['r2'] for m in models]
        axes[1, 0].bar(models, r2_scores, color=['#2E86AB', '#A23B72', '#F18F01'])
        axes[1, 0].set_title('Temporal Forecasting R²')
        axes[1, 0].set_ylabel('R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MSE comparison (forecasting) - lower is better
        mse_scores = [forecasting_results[m]['mse'] for m in models]
        axes[1, 1].bar(models, mse_scores, color=['#2E86AB', '#A23B72', '#F18F01'])
        axes[1, 1].set_title('Temporal Forecasting MSE (lower is better)')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("plots/performance_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print summary table
        print("\nSUMMARY RESULTS:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'F1':<10} {'R²':<10} {'MSE':<10}")
        print("-" * 80)
        
        for model in models:
            acc = classification_results[model]['accuracy']
            f1 = classification_results[model]['f1']
            r2 = forecasting_results[model]['r2']
            mse = forecasting_results[model]['mse']
            print(f"{model:<20} {acc:<10.4f} {f1:<10.4f} {r2:<10.4f} {mse:<10.6f}")
        
        # Highlight best performer
        best_acc_model = max(models, key=lambda m: classification_results[m]['accuracy'])
        best_r2_model = max(models, key=lambda m: forecasting_results[m]['r2'])
        
        print("-" * 80)
        print(f"🏆 Best Classification: {best_acc_model}")
        print(f"🏆 Best Forecasting: {best_r2_model}")
        
        print("✓ Saved performance comparison to plots/performance_comparison.png")


def load_custom_physics_data(embeddings_dir=None):
    """Load your existing custom physics embeddings and metadata"""
    
    # Auto-detect embedding directory if not specified
    if embeddings_dir is None:
        potential_dirs = [
            "Embeddings_Val_Full",    # Full validation set (preferred)
            "Embeddings_Val",         # Limited validation set  
            "Embeddings_Train",       # Training set fallback
            "Embeddings"              # Original 200-sample set
        ]
        
        embeddings_dir = None
        for dir_name in potential_dirs:
            emb_path = f"{dir_name}/physics_embeddings_layer_11.npy"
            if os.path.exists(emb_path):
                embeddings_dir = dir_name
                print(f"✓ Auto-detected embeddings in: {embeddings_dir}")
                break
        
        if embeddings_dir is None:
            print("❌ No embeddings found in any expected directory!")
            print("Checked:")
            for dir_name in potential_dirs:
                print(f"  - {dir_name}/physics_embeddings_layer_11.npy")
            return None, None, None, None
    
    try:
        # Load embeddings and labels
        custom_embeddings = np.load(f"{embeddings_dir}/physics_embeddings_layer_11.npy")
        labels = np.load(f"{embeddings_dir}/physics_labels.npy")
        
        print(f"✓ Loaded embeddings from: {embeddings_dir}")
        print(f"✓ Embedding shape: {custom_embeddings.shape}")
        print(f"✓ Labels shape: {labels.shape}")
        print(f"✓ Classes: {np.unique(labels)}")
        
        # Determine data source type
        if "Val" in embeddings_dir:
            data_source = "validation"
        elif "Train" in embeddings_dir:
            data_source = "training" 
        else:
            data_source = "unknown"
        
        print(f"✓ Data source: {data_source}")
        
        # Check for multiple metadata sources
        metadata_sources = [
            f"{embeddings_dir}/validation_metadata.csv",  # Direct extractor metadata
            f"{embeddings_dir}/embedding_metadata.csv",   # Unified extractor metadata
            "/cloud_psc_homedir/jessicae-cloud/Jessica/Images/multi_physics_vit/metadata.csv"  # Original metadata
        ]
        
        metadata_df = None
        metadata_source = None
        
        for metadata_path in metadata_sources:
            if os.path.exists(metadata_path):
                metadata_df = pd.read_csv(metadata_path)
                metadata_source = metadata_path
                print(f"✓ Using metadata from: {os.path.basename(metadata_path)}")
                break
        
        if metadata_df is None:
            print("❌ No metadata file found!")
            return None, None, None, None
        
        # Create image paths
        data_dir = "/cloud_psc_homedir/jessicae-cloud/Jessica/Images/multi_physics_vit"
        
        if 'image_path' in metadata_df.columns:
            # Use existing image paths
            image_paths = [os.path.join(data_dir, path) for path in metadata_df['image_path'].head(len(custom_embeddings))]
        else:
            # Fall back to original metadata
            original_metadata = pd.read_csv("/cloud_psc_homedir/jessicae-cloud/Jessica/Images/multi_physics_vit/metadata.csv")
            image_paths = [os.path.join(data_dir, path) for path in original_metadata['image_path'].head(len(custom_embeddings))]
            metadata_df = original_metadata.head(len(custom_embeddings))
        
        # Verify images exist
        valid_indices = []
        valid_paths = []
        missing_count = 0
        
        for i, path in enumerate(image_paths):
            if os.path.exists(path):
                valid_indices.append(i)
                valid_paths.append(path)
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"⚠️  {missing_count} missing images, using {len(valid_paths)} valid images")
            custom_embeddings = custom_embeddings[valid_indices]
            labels = labels[valid_indices]
            metadata_df = metadata_df.iloc[valid_indices].reset_index(drop=True)
            image_paths = valid_paths
        
        print(f"✓ Final dataset: {len(custom_embeddings)} samples")
        
        # Check temporal information
        has_temporal_info = ('trajectory_id' in metadata_df.columns and 
                           'timestep' in metadata_df.columns)
        
        if has_temporal_info:
            trajectories = metadata_df['trajectory_id'].nunique()
            print(f"✓ Found {trajectories} unique trajectories for temporal forecasting")
            
            # Show class distribution
            if 'domain' in metadata_df.columns:
                class_counts = metadata_df['domain'].value_counts()
                print(f"✓ Class distribution:")
                for domain, count in class_counts.head(5).items():
                    print(f"    {domain}: {count} samples")
                if len(class_counts) > 5:
                    print(f"    ... and {len(class_counts) - 5} more classes")
        else:
            print("⚠️  No temporal information - will use simple forecasting")
        
        return custom_embeddings, labels, image_paths, metadata_df
        
    except Exception as e:
        print(f"❌ Error loading custom data: {e}")
        
        # Debug info
        if os.path.exists(embeddings_dir):
            print(f"📁 Files in {embeddings_dir}:")
            for file in os.listdir(embeddings_dir):
                file_path = os.path.join(embeddings_dir, file)
                if file.endswith('.npy'):
                    try:
                        arr = np.load(file_path)
                        print(f"    {file}: shape {arr.shape}")
                    except:
                        print(f"    {file}: (could not load)")
                else:
                    print(f"    {file}")
        
        return None, None, None, None


def main():
    """Run comprehensive physics embedding comparison"""
    print("🔬 Physics Foundation Model vs Baselines Comparison")
    print("="*60)
    
    # Create output directory
    os.makedirs("plots", exist_ok=True)
    
    # Load your custom physics data (auto-detect directory)
    custom_embeddings, labels, image_paths, metadata = load_custom_physics_data()
    
    if custom_embeddings is None:
        print("❌ Cannot load custom physics data")
        print("\n💡 Available options:")
        print("   1. Run embedding extraction:")
        print("      python save_embeddings.py --split val --output_dir Embeddings_Val")
        print("   2. Or use direct extractor for full validation set:")
        print("      python direct_validation_extractor.py --checkpoint ... --output_dir Embeddings_Val_Full")
        return
    
    # Initialize comparator
    comparator = PhysicsEmbeddingComparator(custom_embeddings, labels, image_paths, metadata)
    
    # Extract baseline embeddings (DINO, CLIP)
    print("\n1. Extracting baseline embeddings...")
    comparator.extract_baseline_embeddings()
    
    # Run comparisons
    print("\n2. Running classification comparison...")
    classification_results = comparator.run_classification_comparison()
    
    print("\n3. Running temporal forecasting comparison...")
    forecasting_results = comparator.run_temporal_forecasting_comparison(metadata)
    
    print("\n4. Creating PCA analysis...")
    comparator.create_pca_analysis()
    
    print("\n5. Creating t-SNE visualizations...")
    comparator.create_tsne_comparison()
    
    print("\n6. Creating summary comparison...")
    comparator.create_summary_comparison(classification_results, forecasting_results)
    
    print("\n" + "="*60)
    print("✅ COMPARISON COMPLETE!")
    print("📁 Check plots/ directory for visualizations:")
    print("   - pca_analysis.png: PCA explained variance & projections")
    print("   - pca_pairplot.png: Detailed PCA pairwise plots") 
    print("   - tsne_comparison.png: Class separation visualization") 
    print("   - performance_comparison.png: Metric comparison")
    print("   - confusion_matrix_*.png: Individual confusion matrices")
    print("="*60)


if __name__ == "__main__":
    main()