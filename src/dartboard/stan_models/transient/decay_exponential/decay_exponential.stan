
data {
  int<lower=0> N;
  vector<lower=0>[N] t;
  vector[N] g;
  real mu_J;
  real std_J;
  real mu_sigma;
  real std_sigma;
  real std_A;
  real mu_T;
  real std_T;
}
parameters {
  real Jinf;
  real<lower=0> sigma;
  real<lower=0> T_lambda;
  real A;
}
model {
  real alpha_sigma= pow(mu_sigma, 2)/pow(std_sigma, 2);
  real beta_sigma= mu_sigma/pow(std_sigma, 2);

  real alpha_T= pow(mu_T, 2)/pow(std_T, 2);
  real beta_T= mu_T/pow(std_T, 2);

  Jinf ~ normal(mu_J, std_J);
  sigma ~ gamma(alpha_sigma, beta_sigma);
  T_lambda ~ gamma(alpha_T, beta_T);
  A ~ normal(0.0, std_A);

  for (n in 1:N)
    g[n] ~ normal(Jinf + A*exp(-1.0/T_lambda*t[n]), sigma);
}
