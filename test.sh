echo "Random Forest\n" >> test.txt
python predict.py rf >> test.txt
python predict.py rf >> test.txt
python predict.py rf >> test.txt

echo "\n" >> test.txt

echo "Linear Regression\n" >> test.txt
python predict.py linreg >> test.txt
python predict.py linreg >> test.txt
python predict.py linreg >> test.txt

echo "\n" >> test.txt

echo "Gradient Boosting Regressor\n" >> test.txt
python predict.py gbreg >> test.txt
python predict.py gbreg >> test.txt
python predict.py gbreg >> test.txt

echo "\n" >> test.txt

echo "XGB\\n" >> test.txt
python predict.py xgb >> test.txt
python predict.py xgb >> test.txt
python predict.py xgb >> test.txt


