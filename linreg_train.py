# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linreg_train.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgeruss <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/17 17:14:41 by fgeruss           #+#    #+#              #
#    Updated: 2020/11/17 17:14:49 by fgeruss          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import argparse
import pandas as pd
from matplotlib import pyplot as plt
from linreg.gradient_descent import GradientDescent
from linreg.minmaxscaler import MinMaxScaler
import sys

if __name__ == '__main__':
	if 'data.csv' in sys.argv:
		parser = argparse.ArgumentParser()
		parser.add_argument("dataset", type=str, help="input dataset")
		parser.add_argument("-w", "--weights", action="store_true", help="Weights")
		parser.add_argument("-p", "--plot",  action="store_true", help="Plot")
		parser.add_argument("--rmse",  action="store_true", help="MSE")
		parser.add_argument("--mae",  action="store_true", help="MAE")
		args = parser.parse_args()
	else:
		print("usage: python3 linreg_train.py data.csv")
		exit(0)

	df = pd.read_csv(args.dataset)

	scaler = MinMaxScaler()
	df_scaled = scaler.fit_transform(df.km)

	X = pd.DataFrame(df_scaled)
	y = df['price']

	plot = False
	if args.plot:
		plot = True
	gd = GradientDescent(plot=plot)
	gd.fit(X, y)

	if args.weights:
		print(gd.theta)
	y_pred = gd.predict(X)

	if args.plot:
		plt.figure(2)
		plt.plot(gd.mse.keys(), gd.mse.values(), marker='o', color='blue')
		plt.xlabel('Steps')
		plt.ylabel('MSE')
		plt.grid(True)
		plt.show()

	if args.rmse:
		print("Root mean squared error:", gd.mean_squared_error(y, y_pred) ** 0.5)

	if args.mae:
		print("Mean absolute error:", gd.mae(y, y_pred))

	pd.DataFrame(gd.theta, columns=['theta0', 'theta1']).to_csv("Weights.csv", index=False)
