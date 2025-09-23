import bentoml
import numpy as np

def test_model_runs():
    ref = bentoml.sklearn.get("iris_clf:latest")
    runner = ref.to_runner()
    runner.init_local()
    X = np.array([[5.1, 3.5, 1.4, 0.2]])
    y = runner.predict.run(X)
    assert y.shape == (1,)

