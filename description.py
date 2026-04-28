import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import warnings
import albumentations as A
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings('ignore')

class AugmentedProductDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, aug_transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            with Image.open(self.image_paths[idx]) as img:
                img = img.convert('RGB')
                if max(img.size) > 1000:
                    scale = 1000 / max(img.size)
                    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                    img = img.resize(new_size, Image.LANCZOS)
                image = img.copy()
                label = self.labels[idx]

                if self.aug_transform and np.random.random() < 0.15:
                    try:
                        image_np = np.array(image)
                        image_np = self.aug_transform(image=image_np)['image']
                        image = Image.fromarray(image_np)
                    except Exception as e:
                        print(f"Augmentation error (skipping): {e}")

                if self.transform:
                    image = self.transform(image)

                return image, label

        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            image = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]

def create_augmented_sample(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        aug = A.Compose([
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5),
            ], p=0.3),
        ])
        augmented = aug(image=image_np)['image']
        return Image.fromarray(augmented)
    except Exception as e:
        print(f"Error in augmentation for {image_path}: {e}")
        return Image.open(image_path).convert('RGB')

class ModelLayers(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "MobileNet"):
        super().__init__()

        # load a pretrained backbone and freeze it
        self.base_model, num_features = self._initialize_model(model_name)

        # fresh classifier (trainable by default)
        self.classifier = self._build_classifier(num_features, num_classes)

        # attach the classifier in the right place
        if hasattr(self.base_model, "fc"):              # ResNet-style
            self.base_model.fc = self.classifier
        elif hasattr(self.base_model, "classifier"):    # MobileNet / EfficientNet / VGG
            self.base_model.classifier = self.classifier
        else:
            raise ValueError("Unknown model head for replacement")


    def _initialize_model(self, model_name):
        if model_name == "MobileNet":
            model = models.mobilenet_v2(weights="IMAGENET1K_V1")
            in_feats = model.classifier[1].in_features

        elif model_name == "ResNet":
            model = models.resnet50(weights="IMAGENET1K_V1")
            in_feats = model.fc.in_features

        elif model_name == "EfficientNet":
            model = models.efficientnet_b0(weights="IMAGENET1K_V1")
            in_feats = model.classifier[1].in_features

        elif model_name == "VGG":
            model = models.vgg16(weights="IMAGENET1K_V1")
            in_feats = model.classifier[6].in_features

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # 🔒  Freeze everything
        for p in model.parameters():
            p.requires_grad = False

        return model, in_feats


    def _build_classifier(self, num_features, num_classes):
        return nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def create_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Less aggressive crop
        transforms.RandomHorizontalFlip(p=0.2),  # Lower flip probability
        transforms.RandomRotation(10),  # Smaller rotation angle
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),  # Subtler color changes
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((240, 240)),  # Slightly larger resize
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.02, contrast=0.02),  # Add mild jitter to validation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    aug_transform = A.Compose([
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 20.0)),
            A.GaussianBlur(blur_limit=3),
        ], p=0.1),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5),
        ], p=0.1),
    ])
    return train_transform, val_transform, test_transform, aug_transform

def load_and_prepare_data(min_samples_per_class=12):
    file_path = "test/Product_Final.xlsx"
    if not os.path.exists(file_path):
        raise FileNotFoundError("Product_Final.xlsx not found!")
    product_df = pd.read_excel(file_path, sheet_name="Product")
    image_df = pd.read_excel(file_path, sheet_name="Product_Images")

    data = image_df.merge(
        product_df[['Product ID', 'Product Description', 'Product Category']],
        on='Product ID',
        how='inner'
    )

    image_paths = []
    labels = []
    valid_extensions = ['.png', '.jpg', '.jpeg']
    print("Loading initial images...")
    for _, row in tqdm(data.iterrows()):
        for ext in valid_extensions:
            image_path = os.path.join("images", str(row['Product Category']), row['Image'] + ext)
            if os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        if img.size[0] > 10 and img.size[1] > 10:
                            image_paths.append(image_path)
                            labels.append(str(row['Product Description']))
                            break
                except Exception as e:
                    print(f"Skipping corrupted image {image_path}: {e}")
                    continue

    class_counts = pd.Series(labels).value_counts()
    augmented_dir = "augmented_images"
    augmented_paths = []
    augmented_labels = []

    if os.path.exists(augmented_dir):
        print(f"Found existing augmented images directory: {augmented_dir}")
        for class_label in class_counts.index:
            existing_aug_images = list(Path(augmented_dir).glob(f"{class_label}_*.jpg"))
            for img_path in existing_aug_images:
                augmented_paths.append(str(img_path))
                augmented_labels.append(class_label)
        print(f"Found {len(augmented_paths)} existing augmented images")
    else:
        os.makedirs(augmented_dir, exist_ok=True)
        print(f"Created new augmented images directory: {augmented_dir}")

    print("Checking if additional augmentation is needed...")
    augmentation_multiplier = 3

    all_paths = image_paths + augmented_paths
    all_labels = labels + augmented_labels
    combined_class_counts = pd.Series(all_labels).value_counts()
    classes_needing_augmentation = 0

    for class_label, count in tqdm(class_counts.items()):
        combined_count = combined_class_counts.get(class_label, 0)
        target_count = min_samples_per_class * augmentation_multiplier
        if combined_count < target_count:
            classes_needing_augmentation += 1
            class_images = [path for path, label in zip(image_paths, labels) if label == class_label]
            if not class_images:
                continue
            num_samples_needed = target_count - combined_count
            print(f"Augmenting class '{class_label}': adding {num_samples_needed} images")
            for i in range(num_samples_needed):
                original_image_path = class_images[i % len(class_images)]
                augmented_image = create_augmented_sample(original_image_path)
                save_path = os.path.join(augmented_dir, f"{class_label}_{combined_count + i}.jpg")
                augmented_image.save(save_path)
                all_paths.append(save_path)
                all_labels.append(class_label)

    if classes_needing_augmentation == 0:
        print("No additional augmentation needed. Using existing augmented images.")

    image_paths = all_paths
    labels = all_labels
    class_counts = pd.Series(labels).value_counts()
    valid_classes = class_counts[class_counts >= 2].index.tolist()
    filtered_indices = [i for i, label in enumerate(labels) if label in valid_classes]
    image_paths = [image_paths[i] for i in filtered_indices]
    labels = [labels[i] for i in filtered_indices]

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, encoded_labels,
        test_size=0.2,
        stratify=encoded_labels,
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.2,
        stratify=y_temp,
        random_state=42
    )

    print("\nDataset statistics:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    class_distribution = pd.DataFrame({
        'Class': label_encoder.classes_,
        'Train': [sum(y_train == i) for i in range(len(label_encoder.classes_))],
        'Val': [sum(y_val == i) for i in range(len(label_encoder.classes_))],
        'Test': [sum(y_test == i) for i in range(len(label_encoder.classes_))]
    })
    print("\nClass distribution:")
    print(class_distribution)

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder

def evaluate_model(model, data_loader, criterion, device, phase='val'):
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=f'Evaluating {phase}'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = outputs.max(1)
            predictions.extend(preds.numpy())
            true_labels.extend(labels.numpy())

    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    avg_loss = running_loss / len(data_loader)
    return accuracy, avg_loss, predictions, true_labels

def calculate_metrics(y_true, y_pred, label_encoder=None):
    """Calculate and print detailed evaluation metrics."""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Prepare detailed report
    metrics_summary = {
        'accuracy': accuracy * 100,
        'precision_macro': precision_macro * 100,
        'recall_macro': recall_macro * 100,
        'f1_macro': f1_macro * 100,
        'precision_weighted': precision_weighted * 100,
        'recall_weighted': recall_weighted * 100,
        'f1_weighted': f1_weighted * 100
    }
    
    # Print metrics
    print("\n===== Evaluation Metrics =====")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro-Precision: {precision_macro * 100:.2f}%")
    print(f"Macro-Recall: {recall_macro * 100:.2f}%")
    print(f"Macro-F1 Score: {f1_macro * 100:.2f}%")
    print(f"Weighted-Precision: {precision_weighted * 100:.2f}%")
    print(f"Weighted-Recall: {recall_weighted * 100:.2f}%")
    print(f"Weighted-F1 Score: {f1_weighted * 100:.2f}%")
    
    if label_encoder is not None:
        target_names = label_encoder.classes_
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        print("\n===== Classification Report =====")
        print(report)
        
    #     cm = confusion_matrix(y_true, y_pred)
    #     plt.figure(figsize=(12, 10))
    #     plt.title('Confusion Matrix')
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #                 xticklabels=target_names, yticklabels=target_names)
    #     plt.ylabel('True Label')
    #     plt.xlabel('Predicted Label')
    #     plt.xticks(rotation=90)
    #     plt.yticks(rotation=0)
    #     plt.tight_layout()
    #     plt.savefig('confusion_matrix.png')
    #     plt.close()
        
    return metrics_summary

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100, label_encoder=None):
    best_val_acc = 0.0
    patience = 4
    patience_counter = 0
    history = defaultdict(list)
    warmup_epochs = 5
    max_grad_norm = 1.0

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            lr_scale = min(1., (epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * pg['initial_lr']

        model.train()
        running_loss = 0.0
        train_predictions = []
        train_true_labels = []
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')

        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            _, predicted = outputs.max(1)
            train_predictions.extend(predicted.cpu().numpy())
            train_true_labels.extend(labels.cpu().numpy())
            loss_value = loss.item()
            running_loss += loss_value

            del outputs, loss
            torch.cuda.empty_cache()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_bar.set_postfix({'loss': f'{loss_value:.4f}'})

        train_acc = np.mean(np.array(train_predictions) == np.array(train_true_labels))
        train_loss = running_loss / len(train_loader)
        torch.cuda.empty_cache()

        val_acc, val_loss, _, _ = evaluate_model(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%')
        print(f'patience counter: {patience_counter}/{patience}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_acc': best_val_acc,
                'class_names': label_encoder.classes_.tolist(),  # Add class names here
                'num_classes': len(label_encoder.classes_)       # Add number of classes for clarity
            }, 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == num_epochs-1 or  patience_counter >= patience:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_acc': best_val_acc,
            }, 'final_model.pth')
            print("Final model saved as final_model.pth")

        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
        
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)

    return model, history

def evaluate_test_set(model, test_loader, criterion, device, label_encoder):
    test_acc, test_loss, test_predictions, test_true_labels = evaluate_model(
        model, test_loader, criterion, device, phase='test'
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    metrics = calculate_metrics(test_true_labels, test_predictions, label_encoder)
    
    test_results_df = pd.DataFrame({
        'True_Label': [label_encoder.classes_[label] for label in test_true_labels],
        'Predicted_Label': [label_encoder.classes_[pred] for pred in test_predictions],
        'Correct': np.array(test_true_labels) == np.array(test_predictions)
    })
    test_results_df.to_csv('test_predictions.csv', index=False)
    
    return test_acc, test_loss, metrics

def test_custom_images(model, image_paths, transform, label_encoder, device):
    model.eval()
    results = []
    for img_path in tqdm(image_paths, desc="Testing custom images"):
        try:
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                
                # Get top 3 predictions
                confidences, predicted_indices = torch.topk(prob, k=3, dim=1)
                
                # Convert to lists for easier handling
                confidences = confidences.squeeze().tolist()
                predicted_indices = predicted_indices.squeeze().tolist()
                
                # Get class labels for top 3 predictions
                predicted_labels = label_encoder.inverse_transform(predicted_indices)
                
                # Create result dictionary with top 3 predictions
                result = {
                    'image_path': img_path,
                    'top_predictions': []
                }
                
                # Add each of the top 3 predictions with its confidence
                for i in range(3):
                    result['top_predictions'].append({
                        'rank': i + 1,
                        'label': predicted_labels[i],
                        'confidence': confidences[i]
                    })
                
                # Also include the top prediction in the main level for backward compatibility
                result['predicted_label'] = predicted_labels[0]
                result['confidence'] = confidences[0]
                
                results.append(result)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    return results


def main():
    best_mod = 'best_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_and_prepare_data()
    train_transform, val_transform, test_transform, aug_transform = create_transforms()
    
    num_classes = len(label_encoder.classes_)
    model = ModelLayers(num_classes, 'EfficientNet').to(device)

    if not os.path.exists(best_mod):
        print("No pre-trained model found. Training the model")

        train_dataset = AugmentedProductDataset(X_train, y_train, train_transform, aug_transform)
        val_dataset = AugmentedProductDataset(X_val, y_val, val_transform, None)  
        test_dataset = AugmentedProductDataset(X_test, y_test, test_transform, None)  

        train_loader = DataLoader(
            train_dataset,
            batch_size=24,
            shuffle=True,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            prefetch_factor=6,
            persistent_workers=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=24,
            shuffle=False,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=6,
            persistent_workers=False 
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=24,
            shuffle=False,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=6,
            persistent_workers=False  # Added missing parameter
        )
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=5e-5,
            weight_decay=0.1,
            betas=(0.9, 0.999)
        )

        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        model, history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            num_epochs=50,
            label_encoder=label_encoder
        )
        
        plot_training_history(history)


        
    else:
        print("Loading the best_model.pth")
        _, _, X_test, _, _, y_test, _ = load_and_prepare_data() 
        _, _, test_transform, _ = create_transforms()
        test_dataset = AugmentedProductDataset(X_test, y_test, test_transform, None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=24,
            shuffle=False,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=6,
            persistent_workers=False
        )
    
    checkpoint = torch.load(best_mod, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    test_acc, test_loss, test_metrics = evaluate_test_set(
        model, test_loader, criterion, device, label_encoder
    )
    
    with open('test_results.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        for metric_name, value in test_metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")

    test_image_dir = "test_images"
    if os.path.exists(test_image_dir):
        test_images = [str(p) for p in Path(test_image_dir).glob("*.[jp][pn][g]")]
        if test_images:
            print("\nTesting custom images...")
            test_results = test_custom_images(
                model, test_images, test_transform, label_encoder, device
            )

            print("\nCustom Test Results:")
            for r in test_results:
                print(r)
            
            custom_results_df = pd.DataFrame(test_results)
            custom_results_df.to_csv('custom_test_results.csv', index=False)

if __name__ == "__main__":
    main()