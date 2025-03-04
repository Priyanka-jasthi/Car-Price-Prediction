# Car Price Prediction

## Project Overview
This project predicts the price of cars based on various features like engine type, fuel type, car dimensions, and horsepower. It utilizes machine learning models, including Decision Tree, Random Forest, and Gradient Boosting, with hyperparameter tuning to achieve the best performance.

## Technologies Used
- **Python** (pandas, numpy, matplotlib, seaborn, sklearn, joblib)
- **Machine Learning Models** (DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor)
- **Streamlit** (for deployment)
- **Spyder** (for model training)
- **Anaconda** (as the development environment)

## Project Structure
```
├── dataset/                     # CSV dataset used for training
├── models/                      # Saved trained models
│   ├── car_price_model.pkl      # Best trained model
│   ├── feature_columns.pkl      # Feature list used for training                  
│   ├── data_exploration.ipynb   # Initial data analysis
│   ├── model_training.ipynb     # Model training and evaluation
├── app/                         # Streamlit application
│   ├── Streamlit.py             # Main Streamlit app file
```

##  Dataset
The dataset contains various features, including:
- `fueltype` (gas/diesel)
- `carbody` (sedan, convertible, etc.)
- `horsepower`
- `carlength`, `carwidth`, `wheelbase`
- `enginetype`, `cylindernumber`
- `drivewheel`, `aspiration`
- **Target Variable:** `price`

##  Installation & Setup
###  Clone the Repository
```sh
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction
```

###  Install Dependencies
Create a virtual environment (optional but recommended):
```sh
conda create --name car-price python=3.9
conda activate car-price
```
Install required packages:
```sh
pip install -r requirements.txt
```

###  Run Model Training
Train the model using:
```sh
python model_training.py
```
This will generate `car_price_model.pkl` and `feature_columns.pkl`.

###  Run Streamlit App
```sh
streamlit run app/Streamlit.py
```
Open the link displayed in the terminal to use the app.

##  Model Performance
| Model               | R² Score | MAE  |
|--------------------|---------|------|
| Decision Tree      | 0.91   | 1503.03 |
| Random Forest      | 0.95    | 1261.41 |
| Gradient Boosting  | 0.92    | 1726.89 |

The best-performing model (Random-Forest Regressor) is saved for deployment.

## 🛠 Future Improvements
- Add more car features for better predictions.
- Implement deep learning models.
- Enhance UI for better user experience.

##  Contributing
Feel free to fork the repository and submit a pull request with improvements!


