import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing.dataloader import get_dataloader

# 1. Definir un Modelo Dummy Simple (CNN)
class DummyModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DummyModel, self).__init__()
        # Entrada: (1 canal, Alto, Ancho)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # Para forzar una salida fija (Batch, 32, 4, 4) sin importar resolución
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    # 2. Configurar Hiperparámetros y Rutas
    metadata_path = "DMID_PNG/Metadata.xlsx"
    image_dir = "DMID_PNG/1024/TIFF_PREPROCESSED/"
    batch_size = 4
    num_epochs = 2
    learning_rate = 0.001
    num_classes = 2 # Ajustar según el número real de tus clases

    print(f"Iniciando configuración de entrenamiento dummy...")
    
    # Comprobar si hay GPU disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de entrenamiento: {device}")

    # 3. Inicializar el DataLoader
    try:
        dataloader = get_dataloader(
            metadata_path=metadata_path,
            image_dir=image_dir,
            batch_size=batch_size,
            shuffle=True
        )
        print(f"DataLoader inicializado con {len(dataloader.dataset)} imágenes.")
    except Exception as e:
        print(f"Error al inicializar el DataLoader: {e}")
        print("Recuerda: puede que necesites ejecutar 'pip install openpyxl' y verificar el nombre de columnas.")
        return

    # 4. Inicializar Modelo, Función de Pérdida y Optimizador
    num_classes = len(dataloader.dataset.classes)
    model = DummyModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Bucle Principal de Entrenamiento
    print("Iniciando entrenamiento...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            # Mover tensores al dispositivo (GPU/CPU)
            images = images.to(device)
            labels = labels.to(device)

            # Zero los gradientes
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # Calcular pérdida
            loss = criterion(outputs, labels)
            
            # Backward y optimizar
            loss.backward()
            optimizer.step()

            # Estadísticas
            running_loss += loss.item()
            
            if batch_idx % 2 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Media del loss en la epoch
        epoch_loss = running_loss / len(dataloader)
        print(f"==> Epoch {epoch+1} completada. Loss promedio: {epoch_loss:.4f} <==")

    print("Entrenamiento Dummy finalizado con éxito.")

if __name__ == "__main__":
    main()
