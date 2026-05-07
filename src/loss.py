import torch.optim as optim
from torchvision import models
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Spinning up Advanced Training Engine on: {device}")

teacher = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT').to(device)
teacher.eval() 

student = AgroTinyNet(num_classes=3).to(device)
optimizer = optim.AdamW(student.parameters(), lr=0.005, weight_decay=1e-4) # Higher initial LR

# A Learning Rate Scheduler is a huge flex on a resume. 
# It smoothly lowers the learning rate as training progresses.
epochs = 100 
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# We crank up gamma to 3.0. This tells the Focal Loss to aggressively 
# punish the model if it ignores the rare weed/crop pixels.
criterion = AgroImbalanceLoss(alpha=0.5, gamma=3.0)

loss_history = []
print(f"\n🚀 Commencing 50-Epoch Distillation Loop...")

for epoch in range(epochs):
    student.train()
    epoch_loss = 0.0
    start_time = time.time()
    
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        
        with torch.no_grad():
            teacher_logits = teacher(images)['out'][:, :3, :, :] 
            
        student_logits, student_features = student(images)
        
        task_loss = criterion(student_logits, masks)
        kd_loss = torch.nn.functional.mse_loss(student_logits, teacher_logits)
        
        # Shifted focus: 80% on actual GT masks, 20% on teacher hints
        total_loss = (0.8 * task_loss) + (0.2 * kd_loss) 
        
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()
    
    # Step the scheduler to lower the learning rate slightly
    scheduler.step()
        
    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    
    # Only print every 5 epochs to keep the output clean
    if (epoch + 1) % 5 == 0 or epoch == 0:
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

print("✅ Training Complete!")

plt.figure(figsize=(6,3))
plt.plot(loss_history, color='red', marker='.')
plt.title("Advanced Distillation Convergence")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()