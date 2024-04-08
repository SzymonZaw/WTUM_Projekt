import os
import random
import shutil
import torch
import torch.nn as nn
from torchvision import transforms as T, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gradio as gr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ścieżka do danych
data_path = "raw-img"
labels = os.listdir(data_path)
NUM_CLASSES = len(labels)

# Definicja nowych nazw katalogów głównych
directories = ['train', 'val', 'test']

# Definicja nazw podkatalogów dla każdego katalogu głównego
subdirectories = labels

# Proporcje zbioru treningowego, walidacyjnego i testowego
props = [0.95, 0.04, 0.01]

# Przechodzimy przez katalogi główne i podkatalogi, tworząc odpowiednie katalogi
for directory in directories:
    for subdirectory in subdirectories:
        path = os.path.join(data_path, subdirectory)
        files = os.listdir(path)
        random.shuffle(files)
        total = len(files)
        train_end = int(total * props[0])
        val_end = train_end + int(total * props[1])
        if directory == 'train':
            new_files = files[:train_end]
        elif directory == 'val':
            new_files = files[train_end:val_end]
        else:
            new_files = files[val_end:]
        new_path = os.path.join(directory, subdirectory)
        os.makedirs(new_path, exist_ok=True)
        for file in new_files:
            old_file_path = os.path.join(path, file)
            new_file_path = os.path.join(new_path, file)
            shutil.copy(old_file_path, new_file_path)

# Funkcje do ładowania danych
def get_train_valid_loader(train_dir,
                           val_dir,
                           batch_size,
                           augment=True,
                           resize_shape=(224, 224),
                           random_seed=42,
                           shuffle=True):
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Definicja transformacji
    valid_transform = T.Compose([
        T.Resize(resize_shape),
        T.ToTensor(),
        normalize,
    ])
    if augment:
        train_transform = T.Compose([
            T.Resize(resize_shape),
            T.RandomRotation(degrees=(0, 20)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.ToTensor(),
            normalize,
        ])
    else:
        train_transform = T.Compose([
            T.Resize(resize_shape),
            T.ToTensor(),
            normalize,
        ])

    # Ładowanie zbiorów danych
    train_dataset = datasets.ImageFolder(
        root=train_dir, transform=train_transform
    )

    valid_dataset = datasets.ImageFolder(
        root=val_dir, transform=valid_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=shuffle)

    label_mapping = train_dataset.class_to_idx

    return (train_loader, len(train_dataset), valid_loader, len(valid_dataset), label_mapping)


def get_test_loader(test_dir,
                    batch_size,
                    resize_shape=(384, 384),
                    random_seed=42,
                    shuffle=True):
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Definicja transformacji
    test_transform = T.Compose([
        T.Resize(resize_shape),
        T.ToTensor(),
        normalize,
    ])
    test_dataset = datasets.ImageFolder(
        root=test_dir, transform=test_transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle)

    return (test_loader, len(test_dataset))

# Funkcja pomocnicza do pobierania nazwy klasy na podstawie indeksu
def get_className(class_index):
    return list(label_mapping.keys())[list(label_mapping.values()).index(class_index)]

# Funkcja do treningu i walidacji modelu
def train_and_validate_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=1):
    best_valid_loss = float('inf')
    best_valid_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Tryb treningowy i walidacyjny
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Tryb treningowy
                data_loader = train_loader
                total_samples = len(train_loader.dataset)
            else:
                model.eval()   # Tryb walidacyjny
                data_loader = valid_loader
                total_samples = len(valid_loader.dataset)

            running_loss = 0.0
            running_corrects = 0

            # Iteracja po danych
            for i, (inputs, labels) in enumerate(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Wyzerowanie gradientów
                optimizer.zero_grad()

                # Przekazanie przez model
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Obliczanie gradientów i aktualizacja wag tylko w fazie treningowej
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statystyki
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Wyświetlanie postępu treningu
                if phase == 'train':
                    progress = (i + 1) * len(inputs)
                    print(f'Progress: {progress}/{total_samples} ({(progress / total_samples) * 100:.2f}%)', end='\r')

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Zapisanie modelu, jeśli uzyskano lepsze wyniki na zbiorze walidacyjnym
            if phase == 'val' and epoch_loss < best_valid_loss:
                best_valid_loss = epoch_loss
                best_valid_acc = epoch_acc
                #torch.save(model.state_dict(), 'best_model.pth')

    print(f'Best validation loss: {best_valid_loss:.4f}, Best validation accuracy: {best_valid_acc:.4f}')

# Funkcja do strojenia hiperparametrów
def hyperparameter_tuning(model, criterion, train_loader, valid_loader, learning_rates, momentums):
    best_accuracy = 0.0
    best_lr = 0.0
    best_momentum = 0.0

    for lr in learning_rates:
        for momentum in momentums:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
            _, epoch_acc = train_and_validate_model(model, criterion, optimizer, train_loader, valid_loader)

            # Porównanie z najlepszą dokładnością dotychczasową
            if epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_lr = lr
                best_momentum = momentum

    print(f"Best Accuracy: {best_accuracy}")
    print(f"Best Learning Rate: {best_lr}")
    print(f"Best Momentum: {best_momentum}")

# Wczytanie danych treningowych, walidacyjnych i testowych
train_loader, num_train, valid_loader, num_valid, label_mapping = get_train_valid_loader(train_dir='train', val_dir="val", batch_size=32,
                                                                                           augment=True, random_seed=1)
test_loader, num_test = get_test_loader(test_dir="test", batch_size=32)

# Model EfficientNet
model_efficientnet = models.efficientnet_v2_l(weights="IMAGENET1K_V1").to(device)
for param in model_efficientnet.parameters():
    param.requires_grad = False
num_ftrs = model_efficientnet.classifier[1].in_features
model_efficientnet.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
model_efficientnet.to(device)

criterion_efficientnet = nn.CrossEntropyLoss()
optimizer_efficientnet = torch.optim.Adam(model_efficientnet.parameters(), lr=0.001)

# Trenowanie i walidacja modelu EfficientNet
train_and_validate_model(model_efficientnet, criterion_efficientnet, optimizer_efficientnet, train_loader, valid_loader)
torch.save(model_efficientnet.state_dict(), 'best_model_efficientnet.pth')

# Model ResNet-50
model_resnet = models.resnet50(pretrained=True)
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_resnet.to(device)

criterion_resnet = nn.CrossEntropyLoss()
optimizer_resnet = torch.optim.Adam(model_resnet.parameters(), lr=0.001)

# Trenowanie i walidacja modelu ResNet-50
train_and_validate_model(model_resnet, criterion_resnet, optimizer_resnet, train_loader, valid_loader)
torch.save(model_resnet.state_dict(), 'best_model_resnet.pth')
# Eksploracyjna analiza danych

# Wyświetlenie przykładowych obrazów
def display_sample_images(data_loader):
    # Pobranie jednej wsadki (batcha) danych
    images, labels = next(iter(data_loader))
    # Wyświetlenie pierwszych 5 obrazów
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(np.transpose(images[i], (1, 2, 0)))
        ax.set_title(f"Class: {get_className(labels[i].item())}")
        ax.axis('off')
    plt.show()

# Analiza rozkładu klas
def class_distribution(data_loader):
    class_counts = np.zeros(NUM_CLASSES)
    for _, labels in data_loader:
        for label in labels:
            class_counts[label] += 1
    class_names = [get_className(i) for i in range(NUM_CLASSES)]
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, class_counts)
    plt.title('Rozkład klas')
    plt.xlabel('Klasa')
    plt.ylabel('Liczba obrazów')
    plt.xticks(rotation=45)
    plt.show()

# Analiza wielkości obrazów
def image_size_distribution(data_loader):
    image_sizes = []
    for images, _ in data_loader:
        for image in images:
            image_sizes.append(image.shape[-1])  # Rozmiar ostatniej współrzędnej, co odpowiada rozmiarowi wysokości/ szerokości
    plt.figure(figsize=(10, 5))
    plt.hist(image_sizes, bins=20)
    plt.title('Rozkład wielkości obrazów')
    plt.xlabel('Wymiary obrazów')
    plt.ylabel('Liczba obrazów')
    plt.show()

# Funkcje oceny modeli

def evaluate_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return accuracy, precision, recall, f1

# Eksploracyjna analiza danych
print("Przykładowe obrazy:")
display_sample_images(train_loader)

print("Rozkład klas w zbiorze treningowym:")
class_distribution(train_loader)

print("Rozkład wielkości obrazów w zbiorze treningowym:")
image_size_distribution(train_loader)

# Ocena modeli
accuracy_efficientnet, precision_efficientnet, recall_efficientnet, f1_efficientnet = evaluate_model(model_efficientnet, test_loader)
accuracy_resnet, precision_resnet, recall_resnet, f1_resnet = evaluate_model(model_resnet, test_loader)

print("Ocena modelu EfficientNet:")
print(f"Accuracy: {accuracy_efficientnet}, Precision: {precision_efficientnet}, Recall: {recall_efficientnet}, F1: {f1_efficientnet}")

print("Ocena modelu ResNet-50:")
print(f"Accuracy: {accuracy_resnet}, Precision: {precision_resnet}, Recall: {recall_resnet}, F1: {f1_resnet}")

'''
# Załaduj przykładowe obrazy
sample_images, _ = next(iter(train_loader))

# Funkcja do przewidywania klasy obrazka za pomocą modelu EfficientNet
def predict_efficientnet(image):
    image = image.reshape((1, 3, 224, 224))  # Dostosuj kształt obrazu do wejścia modelu
    image = torch.tensor(image).to(device)
    model_efficientnet.eval()
    with torch.no_grad():
        output = model_efficientnet(image)
        _, predicted = torch.max(output, 1)
    return get_className(predicted.item())

# Funkcja do przewidywania klasy obrazka za pomocą modelu ResNet-50
def predict_resnet(image):
    image = image.reshape((1, 3, 224, 224))  # Dostosuj kształt obrazu do wejścia modelu
    image = torch.tensor(image).to(device)
    model_resnet.eval()
    with torch.no_grad():
        output = model_resnet(image)
        _, predicted = torch.max(output, 1)
    return get_className(predicted.item())

# Utwórz interfejs Gradio
inputs = gr.inputs.Image(shape=(224, 224))
outputs_efficientnet = gr.outputs.Label(num_top_classes=1, label="EfficientNet Prediction")
outputs_resnet = gr.outputs.Label(num_top_classes=1, label="ResNet-50 Prediction")

gr.Interface(
    [predict_efficientnet, predict_resnet],
    inputs,
    [outputs_efficientnet, outputs_resnet],
    examples=sample_images.numpy().tolist(),
    title="Image Classifier",
    description="Predicts the class of the input image using EfficientNet and ResNet-50 models."
).launch()
'''

