import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import time
import json
import matplotlib.pyplot as plt

def main():
    # 1. تحميل البيانات وتجهيزها
    df = pd.read_csv('data/housing.csv')
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]

    # Standardization
    X_scaled = (X - X.mean()) / X.std()
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # 2. تقسيم البيانات (80% تدريب، 20% اختبار)
    torch.manual_seed(42) # لضمان ثبات النتائج في كل التجارب
    indices = torch.randperm(len(X_tensor))
    split = int(0.8 * len(X_tensor))
    
    train_indices = indices[:split]
    test_indices = indices[split:]

    X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
    X_test, y_test = X_tensor[test_indices], y_tensor[test_indices]

    # 3. تعريف النموذج (جعله مرناً ليقبل عدد خلايا مختلف)
    class HousingModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(5, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        def forward(self, x): return self.net(x)

    # 4. وظيفة التدريب والتقييم
    def run_experiment(lr, hidden_size, epochs):
        model = HousingModel(hidden_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        start_t = time.time()
        for _ in range(epochs):
            preds = model(X_train)
            loss = criterion(preds, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        duration = time.time() - start_t
        
        # تقييم الأداء على بيانات الاختبار (التي لم يراها النموذج)
        with torch.no_grad():
            test_preds = model(X_test).numpy().flatten()
            test_actual = y_test.numpy().flatten()
            
            # حساب MAE (متوسط الخطأ بالدنانير)
            mae = np.mean(np.abs(test_actual - test_preds))
            
            # حساب R2 (مدى دقة النموذج 0-1)
            ss_res = np.sum((test_actual - test_preds) ** 2)
            ss_tot = np.sum((test_actual - np.mean(test_actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
        return float(mae), float(r2), duration

    # 5. شبكة البحث (Grid Search) - سنجرب 27 تركيبة
    learning_rates = [0.1, 0.05, 0.01]
    hidden_sizes = [16, 32, 64]
    epochs_list = [100, 200, 300]

    all_results = []
    print(f"{'LR':<6} | {'Hidden':<6} | {'Epochs':<6} | {'MAE':<10} | {'R2':<5}")
    print("-" * 45)

    for lr in learning_rates:
        for hs in hidden_sizes:
            for ep in epochs_list:
                mae, r2, dur = run_experiment(lr, hs, ep)
                all_results.append({
                    "lr": lr, "hidden_size": hs, "epochs": ep,
                    "test_mae": mae, "test_r2": r2, "time_sec": dur
                })
                print(f"{lr:<6} | {hs:<6} | {ep:<6} | {mae:<10.2f} | {r2:<5.2f}")

    # 6. حفظ النتائج وترتيبها
    all_results.sort(key=lambda x: x['test_mae']) # ترتيب من الأقل خطأ للأكبر
    
    with open('experiments.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    # 7. رسم بياني ملخص
    plt.figure(figsize=(10, 5))
    maes = [res['test_mae'] for res in all_results]
    plt.plot(maes, marker='o', color='b')
    plt.title('MAE for all experiments (Sorted Best to Worst)')
    plt.xlabel('Experiment Number')
    plt.ylabel('Test MAE (JOD)')
    plt.grid(True)
    plt.savefig('experiment_summary.png')
    
    print(f"\nDone! Best MAE found: {all_results[0]['test_mae']:.2f} JOD")
    print("Results saved to experiments.json and experiment_summary.png")

if __name__ == "__main__":
    main()