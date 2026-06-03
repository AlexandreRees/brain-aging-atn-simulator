import numpy as np
import pandas as pd

def generate_synthetic_brain_data(n=500, seed=42):
    np.random.seed(seed)

    age = np.random.normal(72, 8, n)

    logit = -8 + 0.08 * age + np.random.normal(0, 1, n)
    p_ad = 1 / (1 + np.exp(-logit))
    diagnosis = (p_ad > 0.5).astype(int)

    hippocampus = 4200 - 35*age - 800*diagnosis + np.random.normal(0, 150, n)
    ventricles = 3000 + 40*age + 1200*diagnosis + np.random.normal(0, 200, n)
    cortical_thickness = 2.8 - 0.015*age - 0.4*diagnosis + np.random.normal(0, 0.05, n)

    icv = np.random.normal(1500000, 80000, n)

    mmse = (
        30
        - 0.15*(age - 60)
        - 4*diagnosis
        + 0.0005*hippocampus
        - 0.0003*ventricles
        + np.random.normal(0, 1.5, n)
    )

    mmse = np.clip(mmse, 0, 30)

    df = pd.DataFrame({
        "age": age,
        "diagnosis": diagnosis,
        "hippocampus": hippocampus,
        "ventricles": ventricles,
        "cortical_thickness": cortical_thickness,
        "icv": icv,
        "mmse": mmse
    })

    return df


if __name__ == "__main__":
    df = generate_synthetic_brain_data()
    df.to_csv("data/synthetic_dataset.csv", index=False)
    print("Dataset saved:", df.shape)
