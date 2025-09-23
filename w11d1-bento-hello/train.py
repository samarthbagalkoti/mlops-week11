import bentoml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    X, y = load_iris(return_X_y=True, as_frame=False)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, ytr)

    acc = accuracy_score(yte, model.predict(Xte))
    print(f"Test accuracy: {acc:.3f}")

    # Save into BentoML model store with signatures for serving
    bentoml.sklearn.save_model(
        "iris_clf",
        model,
        signatures={
            "predict": {"batchable": True},
            "predict_proba": {"batchable": True},
        },
        metadata={"accuracy": float(acc)},
    )
    print("Saved model as 'iris_clf' in BentoML model store.")

if __name__ == "__main__":
    main()

