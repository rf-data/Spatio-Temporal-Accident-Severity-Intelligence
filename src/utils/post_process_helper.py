## post_process_helper.py
# imports
import numpy as np





def meta_model():

    X_meta = pd.DataFrame({
                "model_1": y_proba_1,
                "model_2": y_proba_2,
                "model_3": y_proba_3
                })

    meta_model = LogisticRegression()
    meta_model.fit(X_meta, y_true)

    y_pred_final = meta_model.predict(X_meta)

    return 