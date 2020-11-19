# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    linreg_testpy                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgeruss <marvin@42.fr>                     +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/11/17 17:14:54 by fgeruss           #+#    #+#              #
#    Updated: 2020/11/17 17:14:56 by fgeruss          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
import pandas as pd
import numpy as np

def input_km():
	print('Please enter a mileage: ')
	try:
		mileage = float(input())
	except ValueError:
		print('EOF on input. Exit...')
		sys.exit(0)
	return mileage

if __name__ == "__main__":
	if len(sys.argv) > 1:
		print("usage: linreg.py")
		exit(0)
		pass

	mileage = input_km()
	if mileage < 0:
		print('Positive, please. Exit...')
		sys.exit(0)

	theta = np.array(pd.read_csv('Weights.csv'))
	print("Predict price =", np.abs(theta[0][0] + (theta[0][1]) * (mileage / 100000.)))