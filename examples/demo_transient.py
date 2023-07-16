import pprint

import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

import dartboard.transient as transient

## specify a synthetic data simulation

# set up the temporal sampling domain
t0 = 0.0
t1 = 100.0
N_span = 500
t_span = np.linspace(t0, t1, N_span + 1)

# normally distributed asymptotic behavior
mean_asy = 5.0
std_asy = 1.0
# plus a transient with significant magnitude
A0_trans = 7.5
# and a characteristic time a significant fraction of the temporal domain
Tlambda_trans = (t1 - t0) / 4.0

## generate the synthetic dataset and plot its mean and noisy sample

ybar_span = mean_asy + A0_trans * np.exp(-t_span / Tlambda_trans)
y_span = ybar_span + std_asy * rng.randn(*t_span.shape)

plt.plot(t_span, ybar_span, "--")
plt.plot(t_span, y_span, ".")

## fit the transient

# initialize the tool
tf = transient.TransientFitter()

# pack up the parameters to facilitate a prior guess
prior_guess = {
    "mu_J": 5.0,
    "std_J": 0.5,
    "mu_T": 60.0,
    "std_T": 10.0,
    "mu_sigma": 1.0,
    "std_sigma": 1.0,
    "std_A": 3.0,
}
# fit using the independent/dependent data and the prior
tf.fit(t_span, y_span, prior=prior_guess)
# extract the results from the cmdstanpy MLE(MAP) result
fit_result = tf.optim.stan_variables()
print("fit complete. results:")
pprint.pprint(fit_result, indent=2)

## get the model function and plot the 95% CI

# the model type stores it's model form, grab it
model_fun = tf.model_type.model_function(fit_result)

# plot with z=0 (mean), and with +/- 1.96 (between them: 95% CI)
p0 = plt.plot(t_span, model_fun(t_span), "-.")
plt.plot(t_span, model_fun(t_span, z=-1.96), ":", c=p0[-1].get_color())
plt.plot(t_span, model_fun(t_span, z=1.96), ":", c=p0[-1].get_color())
plt.show()

#
