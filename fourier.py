#!/usr/bin/python

#####
#Fourier Series Expansion
#####

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Function:
	def __init__(self,func,domain,func_form=None):
		"""
		--- input param --- 
		func_form : string
		domain : list
			[a,b]

		--- attribute --- 
		T : float 
			period of function.
		"""
		self.func = func
		self.func_form = func_form
		self.domain = domain
		self.T = domain[1] - domain[0]

	def plot_func(self):
		plot_x = np.linspace(self.domain[0],self.domain[1],1000)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(plot_x, self.func(plot_x))
		plt.xlabel("x")
		plt.ylabel("y")
		plt.title(self.func_form)
		fig.show()


class Fourier:
	def __init__(self,func_obj,num,basis_num):
		"""
		--- input param --- 
		func_obj : EstimatedFunction object
		num : int
			The number which domain is splitted.
		basis_num : int
			The number of finite series, which mean max index of fourier coefficient.
		
		--- attribute --- 
		L : harf of domain range.
			T = 2L
		"""

		self.T = func_obj.T
		self.L = func_obj.T / 2.
		self.func = func_obj.func
		self.domain = func_obj.domain
		self.num = num
		self.autocor = None
		self.x = np.linspace(func_obj.domain[0],func_obj.domain[1],num)
		self.y = func_obj.func(self.x)
		self.dx = func_obj.T / self.num
		self.basis_num = basis_num
		self.func_hat = None
		self.fourier_coef = None
		self.spectrum = None

	def plot_auto_corr(self):
		def auto_correlation(y,num,k):
			"""
			k : integer 
				the lag of data.
			"""
			zero_vec = np.tile(0,k)
			y_ = y - np.mean(y)
			y1 = np.r_[zero_vec,y_]
			y2 = np.r_[y_,zero_vec]
			return np.dot(y1,y2) / float(num)

		k_num = 100
		k_vec = np.arange(0,k_num,1)
		self.autocor = np.array( [ auto_correlation(self.y,self.num,k) for k in k_vec] )
		plot_k = np.arange(-k_num,k_num,1)
		plot_cor = np.r_[self.autocor[::-1],self.autocor]
		plt.plot(plot_k,plot_cor)

	def __a0_quadrature(self):
		return ( (1./ self.L) * np.sum( self.y * self.dx ) ) / 2.

	def __a_n_quadrature(self, n):
		"""
		n : integer
			the index of coefficient.
		"""
		return (1. / self.L) * np.sum( self.y * np.cos( (n * np.pi / self.L ) * self.x ) * self.dx )

	def __b_n_quadrature(self,n):
		"""
		n : integer
			the index of coefficient.
		"""
		return (1. / self.L) * np.sum( self.y * np.sin( (n * np.pi / self.L) * self.x ) * self.dx )

	def __fourier_each_x(self,x):
		"""
		real value fourie expansion.
		x : float
		"""
		n_vec = np.linspace(1,self.basis_num,self.basis_num)
		a0 = self.__a0_quadrature()
		a_vec = np.array([ self.__a_n_quadrature(n) for n in n_vec ]) #a_1 ~ a_n
		b_vec = np.array([ self.__b_n_quadrature(n) for n in n_vec ]) #b_1 ~ b_n
		cos_vec = np.cos( (n_vec * np.pi / self.L) * x)
		sin_vec = np.sin( (n_vec * np.pi / self.L) * x)
		return a0 + np.dot(a_vec,cos_vec) + np.dot(b_vec,sin_vec)

	def fourier_expansion(self):
		"""
		x_vec : ndarray 
			(n * 1) vector/
			input of estimated fourier expansion function.
		basis_num : integer
			the number of basis.
		"""
		self.func_hat = np.array([ self.__fourier_each_x(x) for x in self.x ])

	def fourier_expansion_plot(self,title):
		"""
		title : string
		"""
		if self.func_hat == None:
			self.fourier_expansion()

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(self.x,self.y,label="original")
		ax.plot(self.x,self.func_hat,label="fourier")
		plt.title(title)
		plt.legend()
		fig.show()

	def __fourier_transform_each_f(self,f):
		"""
		f : float 
			frequency
		"""
		i_vec = np.linspace(self.domain[0], self.domain[1], self.num)
		y = np.sum( self.func(i_vec) * np.exp( - (np.pi * f * i_vec / self.L) * 1j ) )
		return y

	def fourier_transform(self, f_start=0.0, f_end=200):
		#band = (f_end - f_start) / self.num
		#f_vec = np.arange(f_start,f_end,band)
		f_vec = np.arange(f_start,f_end,1.)
		self.fourier_coef = np.array([ self.__fourier_transform_each_f(f) for f in f_vec ])
		self.spectrum = np.abs(self.fourier_coef)

	def spectral_plot(self,normalize_bool=True):
		if self.spectrum == None:
			self.fourier_transform()

		fig = plt.figure()
		ax = fig.add_subplot(111)

		if normalize_bool == True:
			spectrum_norm = self.spectrum / np.max(self.spectrum)
			ax.plot(spectrum_norm)
		else:
			ax.plot(self.spectrum)

		plt.xlabel("frequency")
		plt.ylabel("spectrum")
		plt.title("spectrum plot")
		plt.show()

def g(x):
	return np.sin(x) * np.sin(2.*x) + np.sin(3.*x)

def h(x):
	return 3*x**2 + 5*x + 23

def h2(x):
	return -3*x**2 + 2*x + 10

def h3(x):
	return -2*x**3 + 200*x**2 + 4*x + 4


T = 200.
num = 1000.

g_func_form = "np.sin(x) * np.sin(2.*x) + np.sin(3.*x) + 100"
G = Function(g,[-T/2.,T/2.],g_func_form)
#G.plot_func()
G_fourier = Fourier(G,num=1000,basis_num=100)
G_fourier.fourier_expansion_plot(title="g(x)")
#fourier.plot_auto_corr()
G_fourier.fourier_expansion()
G_fourier.fourier_transform_each_f(4)
G_fourier.fourier_transform(f_start=0,f_end=200)
G_fourier.spectral_plot()


h_func_form = "3*x**2 + 5*x + 23"
H = Function(h,[-T/2.,T/2.],h_func_form)
H_fourier = Fourier(H,num=1000,basis_num=100)
#H_fourier.fourier_expansion()
H_fourier.fourier_expansion_plot(title="h(x)")

h2_func_form = "-3*x**2 + 2*x + 10"
H2 = Function(h2,[-T/2.,T/2.],h2_func_form)
H2_fourier = Fourier(H2,num=1000,basis_num=100)
H2_fourier.fourier_expansion_plot(title="h2(x)")

h3_func_form = "-2*x**3 + 200*x**2 + 4*x + 4"
H3 = Function(h3,[-T/2.,T/2.],h3_func_form)
H3_fourier = Fourier(H3,num=1000,basis_num=100)
H3_fourier.fourier_expansion_plot(title="h3(x)")



"""
N = 1
x = np.linspace(-T, T, num)
y = np.array([ square_wave(i,N) for i in x ])
plt.plot(x,y)

import matplotlib.animation as animation
def fourier_gif(func,save_name,basis_num=None):
	if basis_num == None:
		basis_num = 10

	fig = plt.figure()
	ims = []
	for N in np.linspace(1,basis_num,basis_num):
		print("step : {} / {}".format(N,basis_num))
		func_hat = fourier(x,func,N)
		im = plt.plot(x,func_hat,color="b",label=N)
		ims.append(im)

	ani = animation.ArtistAnimation(fig, ims, interval=100)
	#plt.show()
	ani.save(save_name)

fourier_gif(h3,"h3.mp4",50)
fourier_gif(g,"g.mp4",50)


#reference of gif code.
#http://matsulib.hatenablog.jp/entry/2013/07/15/203419

#####
def square_wave(x,N):
	def sin_func(k,x):
		return np.sin( (2*k-1)*x ) / ( 2*k-1 )
	y = np.sum([ sin_func(k,x) for k in np.linspace(1,N,N) ])
	return y

def plot_gif_square_wave():
	N = 1
	x = np.linspace(-np.pi, np.pi,300)
	y = np.array([ square_wave(i,N) for i in x ])
	plt.plot(x,y)

	import matplotlib.animation as animation
	fig = plt.figure()
	ims = []
	for N in np.linspace(1,50,50):
		y = np.array([ square_wave(i,N) for i in x ])
		im = plt.plot(x,y,color="b",label=N)
		ims.append(im)

	ani = animation.ArtistAnimation(fig, ims, interval=100)
	plt.show()
	ani.save("fourier.mp4")
	import matplotlib
	matplotlib.use("Agg")


plot_gif_square_wave()
"""
