import pandas as pd
from sklearn.model_selection import train_test_split
from data_ingestion import ingest_data
from train_clf_pipeline import train_clf
from train_reg_pipeline import train_reg 
from evaluation_pipeline import evaluate_clf, evaluate_reg 

# Deployment Thresholds
ACCURACY_THRESHOLD = 0.8 
MAE_THRESHOLD = 2.00 

def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()
    
    df = pd.read_csv("ingested/student_placement_merged.csv")


    print("\nStep 2: Data Preparation")

    # Map target column
    mapping = {'Not Placed': 0, 'Placed': 1}
    df['placement_status'] = df['placement_status'].map(mapping)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['placement_status']
    )

    cols_to_drop = ['Student_ID', 'placement_status', 'salary_lpa', 'tenth_percentage', 'twelfth_percentage']

    # Prepare Classification Data
    x_train_clf = train_df.drop(columns=cols_to_drop)
    y_train_clf = train_df['placement_status']
    
    x_test_clf = test_df.drop(columns=cols_to_drop)
    y_test_clf = test_df['placement_status']

    # Prepare Regression Data (Only Take Placed Students)
    placed_mask_train = train_df['placement_status'] == 1
    x_train_reg = train_df[placed_mask_train].drop(columns=cols_to_drop)
    y_train_reg = train_df.loc[placed_mask_train, 'salary_lpa']

    placed_mask_test = test_df['placement_status'] == 1
    x_test_reg = test_df[placed_mask_test].drop(columns=cols_to_drop)
    y_test_reg = test_df.loc[placed_mask_test, 'salary_lpa']

    
    print("\nStep 3: Model Training")
    print("-Training Classifier-")
    run_id_clf = train_clf(x_train_clf, y_train_clf)
    
    print("-Training Regressor-")
    run_id_reg = train_reg(x_train_reg, y_train_reg)


    print("\nStep 4: Model Evaluation")
    
    # These functions will load the models via run_id and evaluate them
    clf_accuracy, clf_f1 = evaluate_clf(x_test_clf, y_test_clf, run_id_clf)
    reg_mae, reg_r2 = evaluate_reg(x_test_reg, y_test_reg, run_id_reg)

    print("\nStep 5: Deployment Decision")
    
    clf_approved = clf_accuracy >= ACCURACY_THRESHOLD
    reg_approved = reg_mae <= MAE_THRESHOLD

    if clf_approved and reg_approved:
        print("SUCCESS: Both models met the thresholds. Approved for deployment!")
    else:
        print("FAILED: One or both models failed to meet the thresholds.")
        print(f"Classifier Approved: {clf_approved} (F1-Score: {clf_accuracy:.4f})")
        print(f"Regressor Approved: {reg_approved} (MAE: {reg_mae:.4f})")

if __name__ == "__main__":
    run_pipeline()