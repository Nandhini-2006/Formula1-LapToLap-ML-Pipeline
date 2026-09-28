
<div align="center">

# FORMULA 1 RACE OUTCOME PREDICTOR

### Machine Learning Pipeline for Predicting Formula 1 Race Finishing Positions

<p>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/FastF1-E10600?style=for-the-badge">
<img src="https://img.shields.io/badge/LightGBM-9ACD32?style=for-the-badge">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
</p>

</div>

---

## DESCRIPTION

**Formula 1 Race Outcome Predictor** is an end-to-end Machine Learning pipeline that predicts Formula 1 driver finishing positions using historical race, timing, driver, constructor, and track-condition data collected through the **FastF1 API**.

The system preprocesses the collected data, transforms categorical driver and team information, normalizes numerical features, and trains two predictive approaches: a **LightGBM Regressor** and a **PyTorch Deep Neural Network**.

To maintain realistic evaluation and prevent data leakage, the pipeline follows a **temporal data-splitting strategy**, using the **2023 Formula 1 season for training** and the **2024 season for testing and evaluation**.

---

## TECHNICAL STACK

<div align="center">

|      Technology     | Role                                      |
| :-----------------: | ----------------------------------------- |
|   **FastF1 API** | Formula 1 race, timing and telemetry data |
|    **Pandas**    | Data processing and manipulation          |
|      **NumPy**    | Numerical computation                     |
|    **LightGBM**   | Gradient Boosting regression              |
|     **PyTorch**   | Deep Neural Network                       |
|  **Scikit-learn** | Preprocessing and ML utilities            |

</div>

---

<div align="center">

<img src="https://media.giphy.com/media/xT9IgG50Fb7Mi0prBC/giphy.gif" width="300" alt="F1 Car">

</div>

</div>
