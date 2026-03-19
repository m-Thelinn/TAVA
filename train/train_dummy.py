import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing.dataloader import get_dataloaders
from metrics.loss import DiceLoss
from metrics.metrics import dice_score, iou_score

# 1. Definir un Modelo Dummy Simple de Segmentación (FCN)
class DummySegModel(nn.Module):
    def __init__(self):
        super(DummySegModel, self).__init__()
        # Entrada: (Batch, 1 canal, Alto, Ancho)
        # Es un modelo muy básico que mantiene la resolución espacial
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # Capa final para predecir 1 canal (probabilidad de máscara, logits)
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.features(x)

def main():
    # 2. Configurar Hiperparámetros
    batch_size = 4
    num_epochs = 2
    learning_rate = 0.001

    print(f"Iniciando configuración de entrenamiento dummy de segmentación 2D...")
    
    # Comprobar si hay GPU disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de entrenamiento: {device}")

    # 3. Inicializar el DataLoader
    try:
        dataloaders = get_dataloaders(
            batch_size=batch_size
        )
        dataloader = dataloaders["train"]
        val_loader = dataloaders["val"]
        print(f"DataLoader train inicializado con {len(dataloader.dataset)} imágenes.")
    except Exception as e:
        print(f"Error al inicializar el DataLoader: {e}")
        print("Asegúrate de que las rutas apuntan a los directorios correctos y contienen las imágenes y máscaras.")
        return

    # 4. Inicializar Modelo, Función de Pérdida y Optimizador
    model = DummySegModel().to(device)
    # Dice Loss para segmentación binaria
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Bucle Principal de Entrenamiento
    print("Iniciando entrenamiento...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            # Mover tensores al dispositivo (GPU/CPU)
            images = images.to(device)
            masks = masks.to(device)

            # Zero los gradientes
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # Calcular pérdida
            loss = criterion(outputs, masks)
            
            # Backward y optimizar
            loss.backward()
            optimizer.step()

            # Estadísticas
            running_loss += loss.item()
            with torch.no_grad():
                running_dice += dice_score(outputs, masks, apply_sigmoid=True).item()
                running_iou += iou_score(outputs, masks, apply_sigmoid=True).item()

        # Media de métricas en la epoch
        epoch_loss = running_loss / len(dataloader)
        epoch_dice = running_dice / len(dataloader)
        epoch_iou = running_iou / len(dataloader)
        print(f"==> Epoch {epoch+1} completada. Loss promedio: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}, IoU: {epoch_iou:.4f} <==")

    print("Entrenamiento Dummy finalizado con éxito.")

if __name__ == "__main__":
    main()
