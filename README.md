## Parkinson's Disease Detection using Machine Learning

### Overview

This project aims to build an accurate machine learning model to predict the presence of Parkinson's Disease in patients based on a comprehensive set of clinical and demographic features. The process involved end-to-end steps of a typical data science pipeline: data loading, exploratory data analysis, feature scaling, training multiple classification models, hyperparameter tuning, and final model selection.

### Dataset

The project utilizes the "Parkinson's_Data.csv" dataset, which contains 2105 patient records across 35 features. The features include:
* **Demographics:** Age, Gender, Ethnicity, Education Level.
* **Lifestyle Factors:** $\text{BMI}$, Smoking, Alcohol Consumption, Physical Activity, Diet Quality, Sleep Quality.
* **Medical History:** Family History of Parkinson's, Traumatic Brain Injury, Hypertension, Diabetes, Depression, Stroke.
* **Clinical Measurements:** Systolic and Diastolic Blood Pressure ($\text{BP}$), various Cholesterol levels ($\text{Total}$, $\text{LDL}$, $\text{HDL}$, $\text{Triglycerides}$).
* **Neurological Scores:** $\text{UPDRS}$ (Unified Parkinson's Disease Rating Scale), $\text{MoCA}$ (Montreal Cognitive Assessment), Functional Assessment.
* **Specific Symptoms (Binary):** Tremor, Rigidity, Bradykinesia, Postural Instability, Speech Problems, Sleep Disorders, Constipation.
* **Target Variable:** **Diagnosis** (0 = No $\text{PD}$, 1 = $\text{PD}$).

### Key Steps and Findings

#### 1. Data Exploration and Preprocessing
* **Data Quality:** The dataset was complete with **no missing or duplicate values**, ensuring high-quality input for modeling.
* **Target Distribution:** The **Diagnosis** column showed an imbalance, with approximately 62% of patients having a positive $\text{PD}$ diagnosis (Class 1) and 38% having a negative diagnosis (Class 0).
* **Feature Analysis ($\text{EDA}$):**
    * **Age:** Patients with a $\text{PD}$ diagnosis tend to be older. The median age for $\text{PD}$ patients appears slightly higher than for non-$\text{PD}$ patients.
    * **Gender:** Diagnosis distribution is relatively similar across genders.
    * **Lifestyle:** There are noticeable differences in the distribution of **BMI**, **Smoking**, **Alcohol Consumption**, **Physical Activity**, **Diet Quality**, and **Sleep Quality** between the two diagnosis groups, suggesting these are relevant features.
    * **Comorbidities:** $\text{EDA}$ of comorbidities like **Family History of Parkinson's**, **Hypertension**, **Depression**, and **Stroke** also showed variations based on the diagnosis.
* **Feature Scaling:** All 15 numerical features were scaled using **StandardScaler** to standardize their range, which is critical for distance-based algorithms like $\text{SVM}$ and beneficial for gradient-based methods.
* **Data Split:** The dataset was split into training and testing sets with a **stratified** approach ($test\_size=0.2$) to maintain the original class distribution in both sets.

#### 2. Model Training and Evaluation (Baseline)
The following classification algorithms were used as baseline models, evaluated on **Accuracy** and **$\text{F1}$-Score** metrics:

| Model | Accuracy | F1-Score (Weighted Avg) |
| :--- | :--- | :--- |
| Logistic Regression | $0.8005$ | $0.80$ |
| Random Forest | $0.9311$ | $0.93$ |
| $\text{SVM}$ (Default) | $0.8361$ | $0.83$ |
| Gradient Boosting | $0.9406$ | $0.94$ |
| $\text{XGBoost}$ | $0.9430$ | $0.94$ |

**XGBoost** and **Gradient Boosting** were the top-performing baseline models, indicating that tree-based ensemble methods are highly effective for this dataset.

#### 3. Hyperparameter Tuning and Final Model

**RandomizedSearchCV** was applied to the top-performing models like XGBoost and Gradient Boosting to optimize their hyperparameters, using AUC {Area Under the ROC Curve} as the primary scoring metric due to its robustness to class imbalance.

| Model | Tuned $\text{AUC}$ Score (on test set) |
| :--- | :--- |
| Gradient Boosting (Tuned) | $0.9817$ |
| **XGBoost (Tuned)** | **$0.9836$** |

The **Tuned XGBoost Classifier** achieved the highest $\text{AUC}$ score, demonstrating superior performance and generalization capability. The final best-performing model and the fitted StandardScaler object were saved using Python's `pickle` library for future deployment.

#### 4. Future Improvements
To further enhance the robustness and performance of this diagnostic model, the following improvements are planned:

Addressing Class Imbalance: Implement advanced sampling techniques, such as SMOTE (Synthetic Minority Over-sampling Technique) on the training data, to balance the target variable's classes. This is expected to improve the model's ability to generalize, particularly for the minority class (Non-PD or Class 0).

Advanced Feature Selection: Employ techniques like Recursive Feature Elimination (RFE) or L1 Regularization (Lasso) to identify and remove redundant or less impactful features. A reduced feature set can lead to faster training, better interpretability, and potentially higher generalization performance.

Deep Learning Implementation: Experiment with a Multi-Layer Perceptron (MLP) Neural Network to explore whether deep learning architectures can capture more complex, non-linear relationships within the features, potentially surpassing the performance of ensemble models.

Model Explainability (XAI): Integrate explainability tools like SHAP (SHapley Additive exPlanations) to provide insights into which features contribute most to the model's predictions. This is crucial for gaining trust and clinical utility in a medical diagnostic context.
