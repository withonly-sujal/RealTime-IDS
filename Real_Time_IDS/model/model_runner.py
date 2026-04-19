import joblib
import numpy as np


class ModelRunner:
    def __init__(self, model_path):
        loaded = joblib.load(model_path)

        self.meta_model = loaded["meta_model"]

        self.xgb_proc = loaded["xgb_proc"]
        self.xgb_sel = loaded["xgb_sel"]

        self.mlp_proc = loaded["mlp_proc"]
        self.mlp_sel = loaded["mlp_sel"]

        self.scaler_proc = loaded["scaler_proc"]
        self.scaler_sel = loaded["scaler_sel"]

        self.features_proc = loaded["features_proc"]
        self.features_sel = loaded["features_sel"]

    def predict(self, df):
        # Split features
        for col in self.features_proc:
            if col not in df:
                df[col] = 0
                
        X_proc = df[self.features_proc]
        
        for col in self.features_sel:
            if col not in df:
                df[col] = 0
        
        X_sel = df[self.features_sel]

        # XGBoost
        xgb_proc_prob = self.xgb_proc.predict_proba(X_proc)[:, 1]
        xgb_sel_prob = self.xgb_sel.predict_proba(X_sel)[:, 1]

        # MLP (with scaling)
        X_proc_scaled = self.scaler_proc.transform(X_proc)
        X_sel_scaled = self.scaler_sel.transform(X_sel)

        mlp_proc_prob = self.mlp_proc.predict(X_proc_scaled).ravel()
        mlp_sel_prob = self.mlp_sel.predict(X_sel_scaled).ravel()

        # Stack
        stack_input = np.column_stack([
            xgb_proc_prob,
            xgb_sel_prob,
            mlp_proc_prob,
            mlp_sel_prob
        ])

        # Final prediction
        y_prob = self.meta_model.predict_proba(stack_input)[:, 1]
        y_pred = (y_prob > 0.7).astype(int)

        return y_pred, y_prob