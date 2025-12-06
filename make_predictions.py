import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"
df = pd.read_csv(FILE_PATH)
y = df["quality"]
X = df.drop(columns=["quality"])

logged_model = "runs:/9fdca13fb0c24105a208bedebdf77988/model"  # Reemplazar <RUN_ID> con el ID del run correspondiente

logged_model = mlflow.pyfunc.load_model(logged_model)

y = logged_model.predict(X)

print(y)
