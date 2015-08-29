Quantifying Property Hazards Before Time of Inspection

https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction

Make sure pip and virtualenv are installed

Setup:

virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

Use Linear Regression: python predict.py linreg
Use Random Forest Regression: python predict.py rf
Use Gradient Boosted Decision Tress (scikit-learn): python predict.py gbreg
Use Gradient Boosted Decision Trees (xgboost): python predict.py xgb

To run tests using all four models and save output to txt file:

bash test.sh
