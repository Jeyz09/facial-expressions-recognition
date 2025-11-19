# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
batch_size = 32
epochs = 10
lr = 0.001

expressions = ['neutral','happy','sad','angry','surprised','fear','disgust','confused','sleepy','excited']
categories = ['human','emoji']

# Data transforms
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = datasets.ImageFolder('data', transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Multi-task CNN
class FER_Emoji_CNN(nn.Module):
    def __init__(self, num_expressions=10):
        super(FER_Emoji_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.category_head = nn.Sequential(
            nn.Linear(128*6*6,128), nn.ReLU(),
            nn.Linear(128,2)  # 0: human, 1: emoji
        )
        self.expression_head = nn.Sequential(
            nn.Linear(128*6*6,128), nn.ReLU(),
            nn.Linear(128,num_expressions)
        )

    def forward(self,x):
        x = self.features(x)
        x_flat = x.view(x.size(0),-1)
        category_out = self.category_head(x_flat)
        expression_out = self.expression_head(x_flat)
        return category_out, expression_out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FER_Emoji_CNN(num_expressions=len(expressions)).to(device)

criterion_category = nn.CrossEntropyLoss()
criterion_expression = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Map folder labels to category and expression
cat_to_idx = { 'human':0, 'emoji':1 }
expr_to_idx = { exp:i for i,exp in enumerate(expressions) }

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_cat = 0
    correct_expr = 0
    total = 0
    for images, labels in train_loader:
        # ImageFolder returns labels 0..num_classes-1; we need category and expression
        batch_size_curr = images.size(0)
        cat_labels = torch.zeros(batch_size_curr, dtype=torch.long)
        expr_labels = torch.zeros(batch_size_curr, dtype=torch.long)
        for i, lbl in enumerate(labels):
            # convert to path string to get category & expression
            path = dataset.samples[lbl][0]
            parts = path.split(os.sep)
            cat_labels[i] = cat_to_idx[parts[-2]]
            expr_labels[i] = expr_to_idx[parts[-1].split('.')[0].split('_')[0]]  # adjust if needed

        images, cat_labels, expr_labels = images.to(device), cat_labels.to(device), expr_labels.to(device)

        optimizer.zero_grad()
        cat_out, expr_out = model(images)
        loss = criterion_category(cat_out, cat_labels) + criterion_expression(expr_out, expr_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, pred_cat = torch.max(cat_out,1)
        _, pred_expr = torch.max(expr_out,1)
        correct_cat += (pred_cat==cat_labels).sum().item()
        correct_expr += (pred_expr==expr_labels).sum().item()
        total += batch_size_curr

    print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(train_loader):.4f} "
          f"Cat Acc: {100*correct_cat/total:.2f}% Expr Acc: {100*correct_expr/total:.2f}%")

# Save model
torch.save(model.state_dict(),'fer_emoji_model.pth')
print("Training finished. Model saved as fer_emoji_model.pth")
