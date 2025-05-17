
import os
import json
import torch
import argparse
from PIL import Image
from datetime import datetime
from torchvision import transforms

class DamageAssessmentModel(torch.nn.Module):
    """Multi-task model for damage severity regression and damage type classification."""
    
    def __init__(self, backbone_name="resnet50", num_damage_types=8):
        super(DamageAssessmentModel, self).__init__()
        
        # Load the backbone model
        if backbone_name == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Identity()  # Remove classification head
        elif backbone_name == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = torch.nn.Identity()  # Remove classification head
        elif backbone_name == "vit_b_16":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads.head = torch.nn.Identity()  # Remove classification head
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Shared features processing
        self.shared_features = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )
        
        # Severity regression head
        self.severity_head = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()  # Constrain output to [0,1]
        )
        
        # Damage type classification head
        self.damage_type_head = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, num_damage_types)
        )
        
        # Classification details head (for specific damage categories like "flood damage", etc.)
        self.classification_details_head = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 3),  # For the 3 categories in the example: flood, severe, moderate
            torch.nn.Softmax(dim=1)  # Output probabilities that sum to 1
        )
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Process through shared layers
        shared = self.shared_features(features)
        
        # Get predictions from each head
        severity = self.severity_head(shared).squeeze()
        damage_types = self.damage_type_head(shared)
        classification_details = self.classification_details_head(shared)
        
        return {
            "severity": severity,
            "damage_types": damage_types,
            "classification_details": classification_details
        }

def load_model(model_path):
    """Load a trained model."""
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    config = checkpoint['config']
    
    # Initialize model with the same configuration
    damage_types = config.get('DAMAGE_TYPES', [
        "structural damage", "flooding", "water damage", "severe damage", 
        "moderate damage", "mild damage", "roof damage", "wall damage"
    ])
    
    model = DamageAssessmentModel(
        backbone_name=config.get('BACKBONE', 'resnet50'),
        num_damage_types=len(damage_types)
    )
    
    # Load saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, damage_types

def run_inference(model, image_path, damage_types, img_size=224):
    """Run inference on a single image."""
    # Prepare image transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Process outputs
    severity = outputs["severity"].item()
    damage_probs = torch.sigmoid(outputs["damage_types"]).numpy()[0]
    classification_details = outputs["classification_details"].numpy()[0]
    
    # Get damage types above threshold
    detected_damage_types = []
    for i, prob in enumerate(damage_probs):
        if prob > 0.5:  # Threshold for positive classification
            detected_damage_types.append(damage_types[i])
    
    # Create classification details dict
    details_dict = {
        "flood damage": float(classification_details[0]),
        "severe building damage": float(classification_details[1]),
        "moderate building damage": float(classification_details[2])
    }
    
    # Create final output format
    result = {
        "Frame_Name": os.path.basename(image_path),
        "Buildings": [
            ["B001", "", "", severity, 1, 1, ", ".join(detected_damage_types)]
        ],
        "Capture date": datetime.now().strftime("%Y-%m-%d"),
        "Region": "",
        "Damage_Assessment": {
            "severity": float(severity),
            "damage_types": detected_damage_types,
            "assessment_date": datetime.now().strftime("%Y-%m-%d"),
            "model_used": "CNN Ensemble",
            "classification_details": details_dict
        }
    }
    
    return result

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Building Damage Assessment Inference')
    parser.add_argument('--image', required=True, help='Path to the input image')
    parser.add_argument('--model', required=True, help='Path to the trained model file')
    parser.add_argument('--output', help='Path to save the output JSON file', default='output.json')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    
    print(f"Loading model from {args.model}...")
    model, config, damage_types = load_model(args.model)
    
    print(f"Running inference on {args.image}...")
    result = run_inference(model, args.image, damage_types, img_size=config.get('IMG_SIZE', 224))
    
    # Save result to JSON file
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"Results saved to {args.output}")
    print(f"Severity score: {result['Damage_Assessment']['severity']:.2f}")
    print(f"Detected damage types: {', '.join(result['Damage_Assessment']['damage_types'])}")

if __name__ == "__main__":
    main()
