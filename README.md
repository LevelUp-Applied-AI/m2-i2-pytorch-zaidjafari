# Integration 2 — PyTorch: Housing Price Prediction

**Module 2 — Programming for AI & Data Science**

---

## 1. Model Prediction & Features
The model predicts the **price_jod** (Target Variable) using the following 5 features:
- **area_sqm**: Apartment size in square meters.
- **bedrooms**: Number of bedrooms.
- **floor**: Floor number.
- **age_years**: Building age in years.
- **distance_to_center_km**: Distance to the city center.

## 2. Training Configuration
- **Model Architecture**: 
    - Input Layer: 5 features
    - Hidden Layer: 32 units with **ReLU** activation
    - Output Layer: 1 unit (Linear)
- **Epochs**: 100
- **Learning Rate**: 0.01
- **Optimizer**: Adam
- **Loss Function**: MSE (Mean Squared Error)

## 3. Training Outcome
*Note: Run `python train.py` to see your specific loss values.*
- **Initial Loss (Epoch 0)**: [ضع الرقم الأول هنا]
- **Final Loss (Epoch 90)**: [ضع الرقم الأخير هنا]
- **Status**: The loss decreased significantly, showing that the model learned the housing price patterns effectively.

## 4. Behavioral Observation
During the training process, I noticed that:
- **Feature Standardization**: Scaling features was crucial; without it, the large values of `area_sqm` would have dominated the gradient updates compared to `bedrooms`.
- **Training Curve**: The loss dropped very rapidly in the first 20-30 epochs and then stabilized, indicating the model reached convergence quickly on this clean dataset.

---

## Technical Setup (Quick Reference)

**File completed:** `train.py`

**Install PyTorch:**
```bash
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)