import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd

# Load the Excel file into a dictionary
def load_product_data(excel_path):
    product_data = pd.read_excel(excel_path)
    product_dict = {
        row['Product Description']: {
            'product_id': row['Product ID'],
            'local': row['Local']
        }
        for _, row in product_data.iterrows()
    }
    return product_dict

# Initialize the product dictionary
product_dict = load_product_data('../../Dataset/Product_Final.xlsx') #../../Dataset/ 

class ModelLayers(nn.Module):
    def __init__(self, num_classes, model_name='MobileNet'):
        super(ModelLayers, self).__init__()
        
        self.base_model, num_features = self._initialize_model(model_name)
        self.classifier = self._build_classifier(num_features, num_classes)
        
        if hasattr(self.base_model, 'fc'):
            self.base_model.fc = self.classifier  # For ResNet
        elif hasattr(self.base_model, 'classifier'):
            self.base_model.classifier = self.classifier 

    def _initialize_model(self, model_name):
        if model_name == 'MobileNet':
            model = models.mobilenet_v2(weights='IMAGENET1K_V1')
            return model, model.classifier[1].in_features
        
        elif model_name == 'ResNet':
            model = models.resnet50(weights='IMAGENET1K_V1')
            return model, model.fc.in_features
        
        elif model_name == 'EfficientNet':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            return model, model.classifier[1].in_features
        
        elif model_name == 'VGG':
            model = models.vgg16(weights='IMAGENET1K_V1')
            return model, model.classifier[6].in_features
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _build_classifier(self, num_features, num_classes):
        """Builds the classifier head used in different models."""
        return nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)


def create_test_transform():
    return transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def predict_image(image_path, model_path='Models/Identification/best_model.pth', model_type='MobileNet', print_results=True):
    # 
    """
    Classify a single image using a trained model
    
    Args:
        model_path (str): Path to the saved model checkpoint
        image_path (str): Path to the image to classify
        model_type (str): Type of model architecture ('MobileNet', 'ResNet', 'EfficientNet', 'VGG')
        print_results (bool): Whether to print the results to console
        
    Returns:
        dict: Dictionary containing prediction results
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if print_results:
        print(f"Using device: {device}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device ,weights_only=False)
    
    # Get class names and number of classes from the checkpoint
    class_names = checkpoint.get('class_names', [])
    num_classes = checkpoint.get('num_classes', len(class_names))
    
    if not class_names and 'class_names' not in checkpoint and print_results:
        print("Warning: Class names not found in checkpoint. Results will show indices instead of labels.")
    
    # Initialize model
    model = ModelLayers(num_classes, model_type).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create transform
    transform = create_test_transform()
    
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Get top 3 predictions (or fewer if num_classes < 3)
            confidences, predicted_indices = torch.topk(probabilities, k=min(3, num_classes), dim=1)
            
            # Convert to lists
            confidences = confidences.squeeze().tolist()
            predicted_indices = predicted_indices.squeeze().tolist()
            
            # Make sure we handle both single value and list
            if not isinstance(confidences, list):
                confidences = [confidences]
                predicted_indices = [predicted_indices]
            
            # Print results if requested
            if print_results:
                print(f"\nTest results for image: {image_path}")
                print(f"Image size: {image.size}")
                print("\nTop predictions:")
                
                for i, (idx, conf) in enumerate(zip(predicted_indices, confidences)):
                    if class_names:
                        label = class_names[idx]
                        print(f"  {i+1}. {label} - Confidence: {conf*100:.2f}%")
                    else:
                        print(f"  {i+1}. Class index {idx} - Confidence: {conf*100:.2f}%")
            
            # Return the results dictionary
            results = {
                'image_path': image_path,
                'top_predictions': []
            }
            
            for i, (idx, conf) in enumerate(zip(predicted_indices, confidences)):
                label = class_names[idx] if class_names else f"Class {idx}"
                product_info = product_dict.get(label, {'product_id': 'Unknown', 'local': 'Unknown'})
                product_id = product_info['product_id']
                local = product_info['local']


                results['top_predictions'].append({
                    'rank': i + 1,
                    'label': label,
                    'confidence': conf,
                    'product_id': product_id,
                    'local': local
                })
                print(results['top_predictions'][i])
            
            results['predicted_label'] = results['top_predictions'][0]['label']
            results['confidence'] = results['top_predictions'][0]['confidence']
            results['product_id'] = results['top_predictions'][0]['product_id']  # Include product ID in the results
            results['local'] = results['top_predictions'][0]['local']
            if results['confidence'] < 0.1:
                results['predicted_label'] = "Couldn't Identify Image"
            return results
        
    except Exception as e:
        error_msg = f"Error processing image {image_path}: {e}"
        if print_results:
            print(error_msg)
        return {'error': error_msg, 'image_path': image_path}

# if __name__ == "__main__":
#     # model_path = 'best_model.pth'
 
#     image_path = 'weird.png'
#     # image_path = imae

#     result = predict_image(image_path)
#     print(result)
#     print("\nTest completed successfully!")
