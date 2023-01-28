source activate py36
mlflow server --backend-store-uri sqlite:///:memory --default-artifact-root s3://mlflow/artifacts/ --serve-artifacts -h 0.0.0.0 -p 8000

