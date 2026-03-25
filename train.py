import pandas as pd
import numpy as np
import torch
import torch.nn as nn

def main():
    # 1. Load Data
    # تأكد أن ملف الـ CSV موجود في مجلد اسمه data
    df = pd.read_csv('data/housing.csv')

    # 2. Separate Features and Target
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]  # نستخدم القوسين المزدوجين للحفاظ على شكل (N, 1)

    # 3. Standardize Features
    # طرح المتوسط والقسمة على الانحراف المعياري
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std

    # 4. Convert to Tensors
    # يجب استخدام float32 لتجنب أخطاء النوع في PyTorch
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    
    print(f"X shape: {X_tensor.shape}")
    print(f"y shape: {y_tensor.shape}")

    # 5. Define the Model
    class HousingModel(nn.Module):
        def __init__(self):
            super().__init__()
            # تعريف الطبقات: 5 مدخلات -> 32 خلية مخفية -> مخرج 1
            self.layer1 = nn.Linear(5, 32)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(32, 1)

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            return x

    # 6. Instantiate Model, Loss, and Optimizer
    model = HousingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 7. Training Loop (100 Epochs)
    num_epochs = 100
    print("\n--- Starting Training ---")
    for epoch in range(num_epochs):
        # Forward pass
        predictions = model(X_tensor)

        # Compute loss
        loss = criterion(predictions, y_tensor)

        # Reset gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

    # 8. Save Predictions
    print("\n--- Generating Predictions ---")
    with torch.no_grad():
        predictions_tensor = model(X_tensor)

    # تحويل التنسورات إلى NumPy لحفظها في ملف CSV
    predictions_np = predictions_tensor.numpy().flatten()
    actuals_np = y_tensor.numpy().flatten()

    results_df = pd.DataFrame({
        'actual': actuals_np,
        'predicted': predictions_np
    })
    
    results_df.to_csv('predictions.csv', index=False)
    print("Saved predictions.csv successfully!")

if __name__ == "__main__":
    main()