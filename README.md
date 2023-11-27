# 🐴 Horse Health Prediction

## Overview

Horse Health Prediction is a Flask-based web app 🌐, Dockerized and powered by the Random Forest algorithm 🌲. It predicts whether a horse will live, die, or be euthanized based on the [Horse Survival Dataset](https://www.kaggle.com/datasets/yasserh/horse-survival-dataset).

## Features

- Flask web app 🚀
- Dockerized environment 🐳
- Random Forest predictions 🌲
- Trained on Kaggle competition dataset 🏆
- Achieved an accuracy of 0.73030 in the private leaderboard 🎯

## Getting Started

### Prerequisites

- Docker 🐋
- Python 3.x 🐍

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/NeeteshC7/Horse_Health_Prediction.git
    cd Horse_Health_Prediction
    ```

2. Build the Docker image:

    ```bash
    docker build -t horse-health-prediction .
    ```

3. Run the Docker container:

    ```bash
    docker run -p 5000:5000 horse-health-prediction
    ```

4. Open your web browser and navigate to [http://localhost:5000](http://localhost:5000) 🌐 to access the Horse Health Prediction web app.

## Usage

1. Input horse features for prediction 🐎.
2. Click the "Predict" button to get the results 📊.

## Kaggle Competition Participation

Participated in [Playground Series - S3E22](https://www.kaggle.com/competitions/playground-series-s3e22/data) Kaggle competition with an accuracy of 0.73030 on the private leaderboard 🏅.

### Kaggle Dataset

Trained on the [Horse Survival Dataset](https://www.kaggle.com/datasets/yasserh/horse-survival-dataset) 📈.

## Model Information

Predictive model built with Random Forest algorithm 🌲, demonstrating superior performance and following ML best practices.

## Acknowledgments

- Kaggle for dataset and hosting the competition 🙌
- Kaggle community for insights and discussions 💬
- [Yasser H.](https://www.kaggle.com/yasserh) for providing the dataset 📦

## Contributing

Contributions welcome! Fork the repository and submit pull requests 🤝.

## Contact

For inquiries or feedback, contact [your.email@example.com](mailto:your.email@example.com) 📧.
