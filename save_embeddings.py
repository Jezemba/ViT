# #!/usr/bin/env python3
# """
# Extract embeddings from trained Physics ViT model
# """

# import os
# import sys
# import torch
# import numpy as np
# import pandas as pd
# from torch.utils.data import DataLoader
# import pickle
# import yaml

# # Add ModelZoo to path
# sys.path.append('/cloud_psc_homedir/jessicae-cloud/modelzoo/src')

# import cerebras.pytorch as cstorch
# from cerebras.modelzoo.models.vision.vision_transformer.model import (
#     VisionTransformerModelConfig, 
#     ViTClassificationWrapperModel
# )
# from cerebras.modelzoo.data.vision.classification.data.physics import (
#     PhysicsProcessor,
#     PhysicsProcessorConfig
# )


# def load_trained_model(checkpoint_path, config_path):
#     """Load trained model from checkpoint"""
#     print(f"Loading model from {checkpoint_path}")
    
#     # Load config
#     with open(config_path, 'r') as f:
#         config_dict = yaml.safe_load(f)
    
#     model_config = VisionTransformerModelConfig(**config_dict['trainer']['init']['model'])
    
#     # Create model
#     model = ViTClassificationWrapperModel(model_config)
    
#     # Load checkpoint using Cerebras method
#     try:
#         # Method 1: Use cstorch.load (Cerebras-specific)
#         state_dict = cstorch.load(checkpoint_path, map_location='cpu')
#         print("Loaded using cstorch.load")
#     except Exception as e:
#         print(f"cstorch.load failed: {e}")
#         try:
#             # Method 2: Direct torch.load with weights_only=True
#             state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
#             print("Loaded using torch.load with weights_only=True")
#         except Exception as e2:
#             print(f"torch.load also failed: {e2}")
#             raise RuntimeError(f"Could not load checkpoint. Errors: cstorch: {e}, torch: {e2}")
    
#     # Handle different checkpoint formats
#     if isinstance(state_dict, dict):
#         if 'model' in state_dict:
#             model.load_state_dict(state_dict['model'])
#         elif 'state_dict' in state_dict:
#             model.load_state_dict(state_dict['state_dict'])
#         else:
#             # Assume the entire dict is the model state
#             model.load_state_dict(state_dict)
#     else:
#         raise RuntimeError(f"Unexpected checkpoint format: {type(state_dict)}")
    
#     model.eval()
#     print("Model loaded successfully!")
#     return model


# def create_dataloader(data_dir, split='val', batch_size=8):
#     """Create dataloader for embedding extraction"""
#     config = PhysicsProcessorConfig(
#         data_processor="PhysicsProcessor",
#         data_dir=data_dir,
#         split=split,
#         batch_size=batch_size,
#         shuffle=False,
#         image_size=[224, 224],
#         transforms=[
#             {"name": "resize", "size": [224, 224]},
#             {"name": "to_tensor"},
#             {"name": "normalize", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
#         ]
#     )
    
#     processor = PhysicsProcessor(config)
#     dataset = processor.create_dataset()
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
#     return dataloader


# def extract_embeddings(model, dataloader, output_dir, extract_layers=None):
#     """
#     Extract embeddings from model
    
#     Args:
#         model: Trained ViT model
#         dataloader: DataLoader for physics data
#         output_dir: Directory to save embeddings
#         extract_layers: List of layer indices to extract from (None for final layer)
#     """
#     if extract_layers is None:
#         extract_layers = [11]  # Final transformer layer
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     all_embeddings = {f'layer_{i}': [] for i in extract_layers}
#     all_embeddings['final_pooled'] = []
#     all_labels = []
#     all_metadata = []
    
#     print(f"Extracting embeddings from {len(dataloader)} batches...")
    
#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(dataloader):
#             print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
            
#             # Get input embeddings (after patch embedding + positional embedding)
#             input_embeddings = model.model.compute_input_embeddings(images)
            
#             # Extract from specified layers
#             for layer_idx in extract_layers:
#                 features = model.model.extract_features(input_embeddings, layer_idx)
#                 # Use CLS token (first token) as representation
#                 cls_embeddings = features[:, 0]  # Shape: (batch_size, hidden_size)
#                 all_embeddings[f'layer_{layer_idx}'].append(cls_embeddings.cpu().numpy())
            
#             # Also get final pooled output
#             _, pooled_output = model.model.vit_model(images)
#             all_embeddings['final_pooled'].append(pooled_output.cpu().numpy())
            
#             # Store labels
#             all_labels.append(labels.cpu().numpy())
            
#             # Store metadata (batch info)
#             batch_metadata = [{'batch_idx': batch_idx, 'sample_idx': i} 
#                              for i in range(len(labels))]
#             all_metadata.extend(batch_metadata)
    
#     # Concatenate all embeddings
#     final_embeddings = {}
#     for key, emb_list in all_embeddings.items():
#         final_embeddings[key] = np.vstack(emb_list)
#         print(f"{key} embeddings shape: {final_embeddings[key].shape}")
    
#     all_labels = np.concatenate(all_labels)
    
#     # Save embeddings
#     print("Saving embeddings...")
#     for key, embeddings in final_embeddings.items():
#         np.save(os.path.join(output_dir, f'physics_embeddings_{key}.npy'), embeddings)
    
#     # Save labels and metadata
#     np.save(os.path.join(output_dir, 'physics_labels.npy'), all_labels)
    
#     # Save metadata as CSV for easier analysis
#     metadata_df = pd.DataFrame(all_metadata)
#     metadata_df['label'] = all_labels
#     metadata_df['class_name'] = metadata_df['label'].map({0: 'pressure', 1: 'velocity_magnitude'})
#     metadata_df.to_csv(os.path.join(output_dir, 'embedding_metadata.csv'), index=False)
    
#     # Save summary
#     summary = {
#         'num_samples': len(all_labels),
#         'embedding_dim': final_embeddings['final_pooled'].shape[1],
#         'layers_extracted': extract_layers,
#         'class_distribution': {int(k): int(v) for k, v in zip(*np.unique(all_labels, return_counts=True))}
#     }
    
#     with open(os.path.join(output_dir, 'embedding_summary.json'), 'w') as f:
#         import json
#         json.dump(summary, f, indent=2)
    
#     print(f"Embeddings saved to {output_dir}")
#     print(f"Summary: {summary}")
    
#     return final_embeddings, all_labels, metadata_df


# def save_model_weights(model, output_path):
#     """Save model weights as pickle file"""
#     model_dict = {
#         'state_dict': model.state_dict(),
#         'model_config': model.model.vit_model.config if hasattr(model.model.vit_model, 'config') else None,
#         'num_classes': model.num_classes
#     }
    
#     with open(output_path, 'wb') as f:
#         pickle.dump(model_dict, f)
    
#     print(f"Model weights saved to {output_path}")


# def main():
#     """Main function to extract embeddings from trained physics ViT"""
    
#     # Paths (adjust these to your actual paths)
#     checkpoint_path = "/cloud_psc_homedir/jessicae-cloud/modelzoo/model_dir/checkpoint_50.mdl"
#     config_path = "/cloud_psc_homedir/jessicae-cloud/modelzoo/src/cerebras/modelzoo/models/vision/vision_transformer/configs/params_physicsmulti_vit.yaml"
#     data_dir = "/cloud_psc_homedir/jessicae-cloud/Jessica/Images/multi_physics_vit"
#     output_dir = "/cloud_psc_homedir/jessicae-cloud/Jessica/Embeddings"
    
#     print("=== Physics ViT Embedding Extraction ===")
    
#     # Check if checkpoint exists
#     if not os.path.exists(checkpoint_path):
#         print(f"ERROR: Checkpoint not found at {checkpoint_path}")
#         print("Available checkpoints:")
#         model_dir = os.path.dirname(checkpoint_path)
#         if os.path.exists(model_dir):
#             for f in sorted(os.listdir(model_dir)):
#                 if f.endswith('.mdl'):
#                     print(f"  {os.path.join(model_dir, f)}")
#         return
    
#     # Load trained model
#     try:
#         model = load_trained_model(checkpoint_path, config_path)
#     except Exception as e:
#         print(f"Failed to load model: {e}")
#         print("\nDebugging info:")
#         print(f"Checkpoint file size: {os.path.getsize(checkpoint_path)} bytes")
        
#         # Try to inspect the checkpoint file
#         try:
#             with open(checkpoint_path, 'rb') as f:
#                 header = f.read(10)
#                 print(f"File header (first 10 bytes): {header}")
#         except Exception as header_e:
#             print(f"Could not read file header: {header_e}")
#         return
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save model weights as pickle
#     save_model_weights(model, os.path.join(output_dir, 'physics_vit_weights.pkl'))
    
#     # Create dataloader (using validation split)
#     try:
#         dataloader = create_dataloader(data_dir, split='val', batch_size=4)
#         print(f"Created dataloader with {len(dataloader)} batches")
#     except Exception as e:
#         print(f"Failed to create dataloader: {e}")
#         return
    
#     # Extract embeddings from multiple layers
#     extract_layers = [6, 9, 11]  # Early, middle, and final transformer layers
#     try:
#         embeddings, labels, metadata = extract_embeddings(
#             model, dataloader, output_dir, extract_layers
#         )
#     except Exception as e:
#         print(f"Failed to extract embeddings: {e}")
#         return
    
#     print("\n=== Extraction Complete ===")
#     print(f"Files saved in: {output_dir}")
#     print("Available files:")
#     if os.path.exists(output_dir):
#         for file in sorted(os.listdir(output_dir)):
#             file_path = os.path.join(output_dir, file)
#             if file.endswith('.npy'):
#                 try:
#                     arr = np.load(file_path)
#                     print(f"  {file}: shape {arr.shape}")
#                 except:
#                     print(f"  {file}: (could not load)")
#             else:
#                 print(f"  {file}")
#     else:
#         print("Output directory not found!")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Unified embedding extraction for Physics ViT models
Handles both small (200 sample) and full temporal models
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pickle
import yaml
import argparse
from pathlib import Path

# Add ModelZoo to path
sys.path.append('/cloud_psc_homedir/jessicae-cloud/modelzoo/src')

import cerebras.pytorch as cstorch
from cerebras.modelzoo.models.vision.vision_transformer.model import (
    VisionTransformerModelConfig, 
    ViTClassificationWrapperModel
)

# Try both processors (old and new)
try:
    from cerebras.modelzoo.data.vision.classification.data.physicsmulti import (
        TemporalMultiPhysicsProcessor,
        TemporalMultiPhysicsProcessorConfig
    )
    TEMPORAL_PROCESSOR_AVAILABLE = True
    print("✓ Temporal processor available")
except ImportError:
    TEMPORAL_PROCESSOR_AVAILABLE = False
    print("⚠️  Temporal processor not available, using basic processor")

try:
    from cerebras.modelzoo.data.vision.classification.data.physics import (
        PhysicsProcessor,
        PhysicsProcessorConfig
    )
    BASIC_PROCESSOR_AVAILABLE = True
    print("✓ Basic processor available")
except ImportError:
    BASIC_PROCESSOR_AVAILABLE = False
    print("❌ Basic processor not available")


class UnifiedEmbeddingExtractor:
    """Unified embedding extractor for different model types"""
    
    def __init__(self, model_type="auto"):
        """
        Args:
            model_type: "small", "full", or "auto" (auto-detect)
        """
        self.model_type = model_type
        self.model = None
        self.config = None
        
    def detect_model_type(self, config_path):
        """Auto-detect model type from config"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        model_config = config_dict['trainer']['init']['model']
        train_config = config_dict['trainer']['fit']['train_dataloader']
        
        # Check processor type
        processor_name = train_config.get('data_processor', 'PhysicsProcessor')
        num_classes = model_config.get('num_classes', 2)
        
        if processor_name == 'TemporalMultiPhysicsProcessor' or num_classes > 6:
            return "full"
        else:
            return "small"
    
    def load_trained_model(self, checkpoint_path, config_path):
        """Load trained model from checkpoint"""
        print(f"Loading model from {checkpoint_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.config = config_dict
        model_config = VisionTransformerModelConfig(**config_dict['trainer']['init']['model'])
        
        # Auto-detect model type if needed
        if self.model_type == "auto":
            self.model_type = self.detect_model_type(config_path)
            print(f"✓ Auto-detected model type: {self.model_type}")
        
        # Create model
        model = ViTClassificationWrapperModel(model_config)
        
        # Load checkpoint using multiple methods
        state_dict = self._load_checkpoint_flexible(checkpoint_path)
        
        # Handle different checkpoint formats
        if isinstance(state_dict, dict):
            if 'model' in state_dict:
                model.load_state_dict(state_dict['model'])
                print("✓ Loaded from 'model' key")
            elif 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
                print("✓ Loaded from 'state_dict' key")
            else:
                # Assume the entire dict is the model state
                model.load_state_dict(state_dict)
                print("✓ Loaded direct state dict")
        else:
            raise RuntimeError(f"Unexpected checkpoint format: {type(state_dict)}")
        
        model.eval()
        self.model = model
        print("✓ Model loaded successfully!")
        return model
    
    def _load_checkpoint_flexible(self, checkpoint_path):
        """Try multiple methods to load checkpoint"""
        errors = []
        
        # Method 1: Cerebras cstorch.load
        try:
            state_dict = cstorch.load(checkpoint_path, map_location='cpu')
            print("✓ Loaded using cstorch.load")
            return state_dict
        except Exception as e:
            errors.append(f"cstorch.load: {e}")
        
        # Method 2: torch.load with weights_only=True
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            print("✓ Loaded using torch.load with weights_only=True")
            return state_dict
        except Exception as e:
            errors.append(f"torch.load weights_only: {e}")
        
        # Method 3: torch.load without weights_only
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            print("✓ Loaded using torch.load without weights_only")
            return state_dict
        except Exception as e:
            errors.append(f"torch.load no weights_only: {e}")
        
        # All methods failed
        raise RuntimeError(f"Could not load checkpoint. Errors: {'; '.join(errors)}")
    
    def create_dataloader(self, data_dir, split='val', batch_size=8):
        """Create appropriate dataloader based on model type"""
        
        if self.model_type == "full" and TEMPORAL_PROCESSOR_AVAILABLE:
            return self._create_temporal_dataloader(data_dir, split, batch_size)
        elif BASIC_PROCESSOR_AVAILABLE:
            return self._create_basic_dataloader(data_dir, split, batch_size)
        else:
            raise RuntimeError("No compatible data processor available")
    
    def _create_temporal_dataloader(self, data_dir, split, batch_size):
        """Create dataloader for temporal/multi-physics model"""
        print(f"Creating temporal dataloader for {self.model_type} model (split: {split})")
        
        # Use validation transforms (no random augmentation)
        transforms = [
            {"name": "expand_to_square", "background_color": [128, 128, 128]},
            {"name": "resize", "size": [224, 224], "interpolation": "bilinear", "antialias": True},
            {"name": "to_tensor"}
            # Note: No normalization in your YAML - preserve physics colormap meanings
        ]
        
        # For evaluation, we want ALL available data, not just balanced samples
        if split == 'val':
            # Use ALL validation trajectories - no artificial limiting!
            max_samples = None  # This should disable the limit
            print("🎯 Using ALL validation data for comprehensive evaluation")
        else:
            # For train split (if needed), still use the training limit
            max_samples = 10
        
        # Create config without max_samples_per_class for validation
        if max_samples is None:
            config = TemporalMultiPhysicsProcessorConfig(
                data_processor="TemporalMultiPhysicsProcessor",
                data_dir=data_dir,
                split=split,
                batch_size=batch_size,
                shuffle=False,
                image_size=[224, 224],
                transforms=transforms,
                use_random_colormap=False  # No augmentation for evaluation
                # No max_samples_per_class - use ALL validation data
            )
        else:
            config = TemporalMultiPhysicsProcessorConfig(
                data_processor="TemporalMultiPhysicsProcessor",
                data_dir=data_dir,
                split=split,
                batch_size=batch_size,
                shuffle=False,
                image_size=[224, 224],
                transforms=transforms,
                max_samples_per_class=max_samples,
                use_random_colormap=False  # No augmentation for evaluation
            )
        
        processor = TemporalMultiPhysicsProcessor(config)
        dataset = processor.create_dataset()
        
        print(f"✓ Created dataset with {len(dataset)} samples")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return dataloader
    
    def _create_basic_dataloader(self, data_dir, split, batch_size):
        """Create dataloader for basic physics model"""
        print(f"Creating basic dataloader for {self.model_type} model")
        
        # Use consistent transforms with your old 200-sample model
        transforms = [
            {"name": "resize", "size": [224, 224]},
            {"name": "to_tensor"},
            {"name": "normalize", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
        ]
        
        config = PhysicsProcessorConfig(
            data_processor="PhysicsProcessor",
            data_dir=data_dir,
            split=split,
            batch_size=batch_size,
            shuffle=False,
            image_size=[224, 224],
            transforms=transforms
        )
        
        processor = PhysicsProcessor(config)
        dataset = processor.create_dataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return dataloader
    
    def extract_embeddings(self, dataloader, output_dir, extract_layers=None):
        """Extract embeddings from model"""
        if extract_layers is None:
            extract_layers = [11]  # Final transformer layer
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_embeddings = {f'layer_{i}': [] for i in extract_layers}
        all_embeddings['final_pooled'] = []
        all_labels = []
        all_metadata = []
        
        print(f"Extracting embeddings from {len(dataloader)} batches...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                print(f"Processing batch {batch_idx + 1}/{len(dataloader)}", end='\r')
                
                # Get input embeddings (after patch embedding + positional embedding)
                input_embeddings = self.model.model.compute_input_embeddings(images)
                
                # Extract from specified layers
                for layer_idx in extract_layers:
                    features = self.model.model.extract_features(input_embeddings, layer_idx)
                    # Use CLS token (first token) as representation
                    cls_embeddings = features[:, 0]  # Shape: (batch_size, hidden_size)
                    all_embeddings[f'layer_{layer_idx}'].append(cls_embeddings.cpu().numpy())
                
                # Also get final pooled output
                _, pooled_output = self.model.model.vit_model(images)
                all_embeddings['final_pooled'].append(pooled_output.cpu().numpy())
                
                # Store labels
                all_labels.append(labels.cpu().numpy())
                
                # Store metadata (batch info)
                batch_metadata = [{'batch_idx': batch_idx, 'sample_idx': i} 
                                 for i in range(len(labels))]
                all_metadata.extend(batch_metadata)
        
        print()  # New line after progress
        
        # Concatenate all embeddings
        final_embeddings = {}
        for key, emb_list in all_embeddings.items():
            final_embeddings[key] = np.vstack(emb_list)
            print(f"✓ {key} embeddings shape: {final_embeddings[key].shape}")
        
        all_labels = np.concatenate(all_labels)
        
        # Save embeddings
        print("Saving embeddings...")
        for key, embeddings in final_embeddings.items():
            np.save(os.path.join(output_dir, f'physics_embeddings_{key}.npy'), embeddings)
        
        # Save labels
        np.save(os.path.join(output_dir, 'physics_labels.npy'), all_labels)
        
        # Enhanced metadata with class mapping
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df['label'] = all_labels
        
        # Create class mapping based on model type
        if self.model_type == "full":
            # 12 physics domains for full model
            class_names = {
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
        else:
            # Field types for small model
            class_names = {0: 'pressure', 1: 'velocity_magnitude'}
        
        metadata_df['class_name'] = metadata_df['label'].map(class_names)
        metadata_df.to_csv(os.path.join(output_dir, 'embedding_metadata.csv'), index=False)
        
        # Save summary
        summary = {
            'model_type': self.model_type,
            'num_samples': len(all_labels),
            'embedding_dim': final_embeddings['final_pooled'].shape[1],
            'layers_extracted': extract_layers,
            'class_distribution': {int(k): int(v) for k, v in zip(*np.unique(all_labels, return_counts=True))},
            'class_names': class_names
        }
        
        import json
        with open(os.path.join(output_dir, 'embedding_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Embeddings saved to {output_dir}")
        print(f"✓ Summary: {summary}")
        
        return final_embeddings, all_labels, metadata_df
    
    def save_model_weights(self, output_path):
        """Save model weights as pickle file"""
        model_dict = {
            'state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.model.num_classes,
            'config': self.config
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_dict, f)
        
        print(f"✓ Model weights saved to {output_path}")


def find_latest_checkpoint(model_dir):
    """Find the latest checkpoint in model directory"""
    if not os.path.exists(model_dir):
        return None
    
    checkpoints = []
    for f in os.listdir(model_dir):
        if f.startswith('checkpoint_') and f.endswith('.mdl'):
            try:
                step = int(f.split('_')[1].split('.')[0])
                checkpoints.append((step, os.path.join(model_dir, f)))
            except:
                continue
    
    if checkpoints:
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]  # Return path of latest checkpoint
    
    return None


def main():
    """Main function to extract embeddings from trained physics ViT"""
    parser = argparse.ArgumentParser(description='Extract embeddings from Physics ViT')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model_dir', type=str, default='/cloud_psc_homedir/jessicae-cloud/modelzoo/model_dir',
                        help='Model directory (for auto-finding latest checkpoint)')
    parser.add_argument('--data_dir', type=str, 
                        default='/cloud_psc_homedir/jessicae-cloud/Jessica/Images/multi_physics_vit',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, 
                        default='/cloud_psc_homedir/jessicae-cloud/Jessica/Embeddings',
                        help='Output directory for embeddings')
    parser.add_argument('--model_type', type=str, choices=['small', 'full', 'auto'], 
                        default='auto', help='Model type')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='train',
                        help='Dataset split to use for embedding extraction')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for extraction')
    parser.add_argument('--layers', type=int, nargs='+', default=[6, 9, 11], 
                        help='Layers to extract embeddings from')
    
    args = parser.parse_args()
    
    print("=== Unified Physics ViT Embedding Extraction ===")
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(args.model_dir)
        if checkpoint_path:
            print(f"✓ Found latest checkpoint: {checkpoint_path}")
        else:
            print(f"❌ No checkpoint found in {args.model_dir}")
            print("Available files:")
            if os.path.exists(args.model_dir):
                for f in os.listdir(args.model_dir):
                    print(f"  {f}")
            return
    
    # Determine config path
    if args.config:
        config_path = args.config
    elif args.model_type == "full" or "temporal" in checkpoint_path.lower():
        config_path = "/cloud_psc_homedir/jessicae-cloud/modelzoo/src/cerebras/modelzoo/models/vision/vision_transformer/configs/params_balanced_temporal_vit.yaml"
    else:
        config_path = "/cloud_psc_homedir/jessicae-cloud/modelzoo/src/cerebras/modelzoo/models/vision/vision_transformer/configs/params_physics_vit.yaml"
    
    print(f"Using config: {config_path}")
    
    # Check if files exist
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return
    
    # Initialize extractor
    extractor = UnifiedEmbeddingExtractor(model_type=args.model_type)
    
    # Load trained model
    try:
        model = extractor.load_trained_model(checkpoint_path, config_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"\nDebugging info:")
        print(f"Checkpoint file size: {os.path.getsize(checkpoint_path)} bytes")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model weights
    extractor.save_model_weights(os.path.join(args.output_dir, 'physics_vit_weights.pkl'))
    
    # Create dataloader
    try:
        dataloader = extractor.create_dataloader(args.data_dir, split=args.split, batch_size=args.batch_size)
        print(f"✓ Created dataloader with {len(dataloader)} batches")
    except Exception as e:
        print(f"❌ Failed to create dataloader: {e}")
        return
    
    # Extract embeddings
    try:
        embeddings, labels, metadata = extractor.extract_embeddings(
            dataloader, args.output_dir, extract_layers=args.layers
        )
    except Exception as e:
        print(f"❌ Failed to extract embeddings: {e}")
        return
    
    print("\n=== Extraction Complete ===")
    print(f"✓ Files saved in: {args.output_dir}")
    print("Available files:")
    for file in sorted(os.listdir(args.output_dir)):
        file_path = os.path.join(args.output_dir, file)
        if file.endswith('.npy'):
            try:
                arr = np.load(file_path)
                print(f"  {file}: shape {arr.shape}")
            except:
                print(f"  {file}: (could not load)")
        else:
            print(f"  {file}")


if __name__ == "__main__":
    main()