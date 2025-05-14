import argparse, pandas as pd, mlflow, mlflow.sklearn, joblib, pathlib

from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss)
from sklearn.base import clone

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from mlflow.models.signature import infer_signature


def setup_mlflow(mode, local_uri=None, repo_owner=None, repo_name=None):
    if mode == 'local':
        if not local_uri:
            raise ValueError("--local_uri harus diisi dalam mode local")
        
        mlflow.set_tracking_uri(local_uri)
        
        print(f"✅ MLflow mode: LOCAL ({local_uri})")
    elif mode == 'online':
        if not (repo_owner and repo_name):
            raise ValueError("--repo_owner dan --repo_name wajib diisi untuk mode online")
        
        import dagshub
        
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        
        print(f"✅ MLflow mode: ONLINE via DagsHub ({repo_owner}/{repo_name})")
    else:
        raise ValueError("Mode harus 'local' atau 'online'")


def tune_and_log(name, pipeline, param_grid, X_train, X_test, y_train, y_test, cv, mode):
    mlflow.set_experiment(f"AllRuns_{name}_{mode}")

    param_combinations = list(ParameterGrid(param_grid))
    best_f1 = -1
    best_model = None
    best_params = None
    best_run_id = None

    for i, params in enumerate(param_combinations):
        pipe = clone(pipeline)
        pipe.set_params(**params)

        with mlflow.start_run(run_name=f"{name}_trial_{i}", nested=True) as run:
            run_id = run.info.run_id
            
            mlflow.log_params(params)

            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
            avg_cv_score = scores.mean()
            
            mlflow.log_metric("cv_f1_weighted", avg_cv_score)

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1", f1)

            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X_test)
                
                mlflow.log_metric("roc_auc_ovr", roc_auc_score(y_test, proba, multi_class='ovr'))
                mlflow.log_metric("log_loss", log_loss(y_test, proba))

            input_example = X_train[:5]
            signature = infer_signature(X_train, pipe.predict(X_train))

            mlflow.sklearn.log_model(pipe, artifact_path="model", signature=signature, input_example=input_example)

            if f1 > best_f1:
                best_f1 = f1
                best_model = pipe
                best_params = params
                best_run_id = run_id

    print(f"\n== {name} ({mode}) Summary ==")
    print(f"Best params: {best_params}")
    print(f"Best Test F1: {best_f1:.3f}")

    return name, best_f1, best_model, best_run_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modelling & Tuning dengan MLflow')
    parser.add_argument('--input', required=True, help='Path ke Obesity_preprocessing.csv')
    parser.add_argument('--mode', choices=['local', 'online'], default='local', help='Pilih mode pelacakan MLflow')
    parser.add_argument('--local_uri', help='URI MLflow lokal (untuk mode local)')
    parser.add_argument('--repo_owner', help='Pemilik repo DagsHub (untuk mode online)')
    parser.add_argument('--repo_name', help='Nama repo DagsHub (untuk mode online)')

    args = parser.parse_args()

    # Setup MLflow
    setup_mlflow(args.mode, args.local_uri, args.repo_owner, args.repo_name)

    # Load preprocessed data
    df = pd.read_csv(args.input)
    X = df.drop('Obesity', axis=1)
    y = df['Obesity']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Model configurations
    configs = [
        ('RandomForest', ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
        ]), {'rf__n_estimators': [100, 200], 'rf__max_depth': [10, 20, None]}),

        ('GradientBoosting', ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42))
        ]), {'gb__n_estimators': [100, 200], 'gb__learning_rate': [0.05, 0.1], 'gb__max_depth': [3, 5]}),

        ('XGBoost', ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('xgb', XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, verbosity=0, random_state=42))
        ]), {'xgb__n_estimators': [100, 200], 'xgb__learning_rate': [0.05, 0.1], 'xgb__max_depth': [3, 5]}),

        ('MLP', ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('mlp', MLPClassifier(max_iter=1000, early_stopping=True, tol=1e-4, random_state=42))
        ]), {'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)], 'mlp__alpha': [1e-4, 1e-3]}),

        ('LogisticRegression', ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs', random_state=42))
        ]), {'lr__C': [0.1, 1.0, 10.0], 'lr__penalty': ['l2']})
    ]

    results = []

    for name, pipe, grid in configs:
        result = tune_and_log(name, pipe, grid, X_train, X_test, y_train, y_test, cv, args.mode)
        results.append(result)

    best_name, best_f1, best_model, best_run_id = max(results, key=lambda x: x[1])
    
    joblib.dump(best_model, "best_model.pkl")

    mlflow.set_experiment(f"BestModel_{best_name}")
    
    with mlflow.start_run(run_name=f"BestModel_{best_name}") as run:
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        conda_env_path = str(pathlib.Path(__file__).parent / "conda.yaml")
        
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train[:5],
            conda_env=conda_env_path
        )
        
        mlflow.log_param("model_name", best_name)
        mlflow.log_metric("f1_weighted", best_f1)
        mlflow.set_tag("best_model", "true")

    print(f"\n🏆 Model terbaik: {best_name} dengan F1-weighted score: {best_f1:.3f}")
    print("📦 Model terbaik disimpan dan ditandai di eksperimen terpisah.")
