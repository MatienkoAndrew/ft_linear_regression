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
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from linreg.gradient_descent import GradientDescent
from linreg.minmaxscaler import MinMaxScaler

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help="input dataset")
	args = parser.parse_args()

	df = pd.read_csv(args.dataset)

	scaler = MinMaxScaler()
	df_scaled = scaler.fit_transform(df.km)

	X = pd.DataFrame(df_scaled)
	y = df['price']
	gd = GradientDescent(plot=False)
	gd.fit(X, y)
	print(gd.theta)
	y_pred = gd.predict(X)
	# print(y_pred)
	# print(gd.mse)
	plt.plot(gd.mse.keys(), gd.mse.values(), marker='o', color='blue')
	plt.show()

	print(gd.mean_squared_error(y, y_pred))
	print(gd.mae(y, y_pred))
