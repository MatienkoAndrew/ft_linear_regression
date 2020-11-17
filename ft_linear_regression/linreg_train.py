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

if __name__ == '__main__':
	parser = argparse.ArgumentParset()
	parser.add_argument("dataset", type=str, help="input dataset")
	args = parser.parse_args()

	df = pd.read_csv('data.csv')
	

