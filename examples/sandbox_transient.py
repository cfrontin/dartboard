import pprint

import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

import dartboard.transient as transient

## specify simulation

t0 = 0.0
t1 = 100.0
N_span = 500
t_span = np.linspace(t0, t1, N_span + 1)

mean_asy = 5.0
std_asy = 1.0
Tlambda_trans = (t1 - t0) / 4.0
A0_trans = 7.5

## generate the synthetic dataset and plot its mean and noisy sample

ybar_span = mean_asy + A0_trans * np.exp(-t_span / Tlambda_trans)
y_span = ybar_span + std_asy * rng.randn(*t_span.shape)

plt.plot(t_span, ybar_span, "--")
plt.plot(t_span, y_span, ".")

## fit the transient

tf = transient.TransientFitter()

prior_guess = {
    "mu_J": 5.0,
    "std_J": 0.5,
    "mu_T": 60.0,
    "std_T": 10.0,
    "mu_sigma": 1.0,
    "std_sigma": 1.0,
    "std_A": 3.0,
}
tf.fit(t_span, y_span, prior=prior_guess)

fit_result = tf.optim.stan_variables()
pprint.pprint(fit_result)

## get the model function and plot the 95% CI

model_fun = tf.model_type.model_function(fit_result)

p0 = plt.plot(t_span, model_fun(t_span), "-.")
plt.plot(t_span, model_fun(t_span, z=-1.96), ":", c=p0[-1].get_color())
plt.plot(t_span, model_fun(t_span, z=1.96), ":", c=p0[-1].get_color())

plt.show()

#
