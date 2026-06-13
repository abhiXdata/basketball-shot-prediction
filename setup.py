from setuptools import setup, find_packages

setup(
    name="basketball-shot-predictor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'Flask==2.3.2',
        'gunicorn==20.1.0',
        'numpy==1.24.3',
        'scikit-learn==1.3.0',
        'joblib==1.3.2',
    ],
)
