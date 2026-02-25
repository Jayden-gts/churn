from xgboost import XGBClassifier

def get_model(scale_pos_weight):
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        # n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        # subsample=0.8,
        eval_metric = 'auc',
        random_state=42
    )