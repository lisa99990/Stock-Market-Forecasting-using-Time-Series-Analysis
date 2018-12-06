# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from django.conf import settings
import matplotlib.pyplot as plt 
import datetime
import numpy
import os

from minor.forms import FileSelectForm

# Create your views here.


def evaluate_sarima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.98)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_sarima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					#print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	true=best_cfg
	#print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
 
def difference(dataset, interval=1):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-1]

def home(request):
	
	file_path = settings.BASE_DIR + '/files_system/'
	df = pd.read_csv(file_path+settings.FILE_TO_USE[0],parse_dates=['Date'],index_col='Date')
	n_df = df[['Close']]
	# use grid search to get best values for p,d,q
	series = pd.Series(n_df.Close,index=n_df.index)
	X = series.values


#stopcode
	size = int(len(series) * 0.98)
	train, test = series[0:size], series[size:len(X)]
	history = [x for x in train]
	predictions = list()
	a=list()
	for t in range(len(test)):
		
		p_values = [0, 1, 2, 4, 6, 8, 10]
		d_values = range(0, 3)
		q_values = range(0, 3)
		#warnings.filterwarnings("ignore")
		model = SARIMAX(history, evaluate_models(series.values, p_values, d_values, q_values))
		model_fit = model.fit(disp=0)
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		a.append(obs)
		print('i=%f,predicted=%f, expected=%f' % (t,yhat, obs))

	#trycode
	#trycode	
# create a differenced series

# 	file_path = settings.BASE_DIR + '/files_system/'
# 	df = pd.read_csv(file_path+settings.FILE_TO_USE[0],parse_dates=['Date'],index_col='Date')
# 	n_df = df[['Close']] 
# # load dataset
# 	series =pd.Series(n_df.Close,index=n_df.index)
# 	# seasonal difference
# 	X = series.values
	d=list()
	days_in_year = 365
	differenced = difference(X, days_in_year)
		# fit model
	model = SARIMAX(differenced, evaluate_models(series.values, p_values, d_values, q_values))
	model_fit = model.fit(disp=0)
		# multi-step out-of-sample forecast
	forecast = model_fit.forecast(steps=5)[0]
		# invert the differenced forecast to something usable
	history = [x for x in X]
	day = 1
	for yhat in range(0,5):
		inverted = inverse_difference(history, yhat, days_in_year)
		print('Day %d: %f' % (day, inverted))
		d.append(inverted)
		history.append(inverted)
		day += 1	

	#stopcode
	# d=model_fit.forecast(steps=5)[0]
	# print(d,'----')
	length=len(test.index)
	conv=test.index.date[length-1]
	b=conv.timetuple()
	b1=b[0]
	b2=b[1]
	b3=b[2]
	dt = datetime.datetime(b1,b2,b3)
	if (b3==30):
		b3=1
		b2=b2+1
	end = datetime.datetime(b1,b2,b3+5)
	step = datetime.timedelta(days=1)
	future_date = []
	while dt < end:
		future_date.append(dt.strftime('%Y-%m-%d'))
		dt += step
	mydir = settings.BASE_DIR+'/media/time_series/'
	ddir = settings.BASE_DIR+'/media/diagnostic/'
	forecast_errors = [test[i]-predictions[i] for i in range(len(predictions))]
	err=numpy.mean(forecast_errors)
	accuracy=100-err
	
	try:
		filelist = [ f for f in os.listdir(mydir) if f.endswith(".png") ]
	except Exception as e:
		print (e)
	else:
		for f in filelist:
			os.remove(os.path.join(mydir, f))



	try:
		filelist = [ f for f in os.listdir(ddir) if f.endswith(".png") ]
	except Exception as e:
		print (e)
	else:
		for f in filelist:
			os.remove(os.path.join(ddir, f))

	d_df = pd.DataFrame({'date':future_date})
	d_df = d_df.set_index(pd.DatetimeIndex(d_df['date']))
	
	fig = plt.figure()
	fig.tight_layout()
	plt.plot(test.index,test.values,label='Observed Values')
	plt.plot(test.index,predictions,label='Predicted Values')

	plt.plot(d_df.index,d,label='Forecasted')
	plt.legend(loc='best')
	plt.grid()
	fig.autofmt_xdate()

	first_image=settings.BASE_DIR+'/media/time_series/'
	fig.savefig(first_image+'fig_2.png')
	# fig = plt.figure()
	#model = SARIMAX(history,evaluate_models(series.values, p_values, d_values, q_values))
	results = model.fit()
	second_image = settings.BASE_DIR + '/media/diagnostic/'

	
	fig = plt.figure()
	results.plot_diagnostics(fig=fig)
	fig.savefig(second_image+'fig_1.png')

	total =dict(zip(a,predictions))
	
	context_dict = {
	'total':total,
	'first_plot': settings.MEDIA_URL+'time_series/fig_2.png',
	'second_plot': settings.MEDIA_URL+'diagnostic/fig_1.png',
	'third_plot': settings.MEDIA_URL+'forecast/fig_3.png',

	}
	return render(request,'home.html',context_dict)

from django.http import HttpResponseRedirect
def get_files(request):
	if request.method=='POST':
		settings.FILE_TO_USE = []
		settings.FILE_TO_USE.append(request.POST['files'])
		return HttpResponseRedirect('/process/')
	else:
		form = FileSelectForm()
	return render(request,'abc.html',{'form':form})

