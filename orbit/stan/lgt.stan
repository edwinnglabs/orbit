// Holt-Winters’ seasonal method
// Additive Trend, Additive Seasonal and Additive Error model
// as known as ETS(A,A,A)
// Hyndman Exponential Smoothing Book page. 46
// Using equation 3.16a-3.16e
// Additional Regression Components are added as r[t]
// normalized seasonal component using chapter 8.1 in initial components
// can consider dynamic normalization later
// lgt version provided an additional global power trend suggested by Slawek

// rr stands for regular regressor(s) where the coef follows normal distribution
// pr stands for positive regressor(s) where the coef follows truncated normal distribution

// --- Code Style for .stan ---
// Upper case for Input
// lower case for intermediate variables and variables we are interested

data {
  // indicator of which method stan using
  int<lower=0,upper=1> WITH_MCMC;
  
  // Data Input
  // Response Data
  int<lower=1> NUM_OF_OBS; // number of observations
  vector<lower=0>[NUM_OF_OBS] RESPONSE;
  // Regression Data
  int<lower=0> NUM_OF_PR; // number of positive regressors
  matrix[NUM_OF_OBS, NUM_OF_PR] PR_MAT; // positive coef regressors, less volatile range
  vector<lower=0>[NUM_OF_PR] PR_BETA_PRIOR;
  vector<lower=0>[NUM_OF_PR] PR_SIGMA_PRIOR;
  int<lower=0> NUM_OF_RR; // number of regular regressors
  matrix[NUM_OF_OBS, NUM_OF_RR] RR_MAT; // regular coef regressors, more volatile range
  vector[NUM_OF_RR] RR_BETA_PRIOR;
  vector<lower=0>[NUM_OF_RR] RR_SIGMA_PRIOR;
  
  // Regression Hyper Params
  // 0 As Fixed Ridge Penalty, 1 As Lasso, 2 As Auto-Ridge
  int <lower=0,upper=2> REG_PENALTY_TYPE;
  real<lower=0> AUTO_RIDGE_SCALE;
  real<lower=0> LASSO_SCALE;
  // Test penalty scale parameter to avoid regression and smoothing over-mixed
  real<lower=0.0> R_SQUARED_PENALTY;
  
  // Trend Hyper-Params
  real<lower=0> LEV_SM_ALPHA;
  real<lower=0> SLP_SM_ALPHA;
  real<lower=0,upper=1> LEV_SM_MAX;
  real<lower=0,upper=1> SLP_SM_MAX;

  // Residuals Tuning Hyper-Params
  // this re-parameterization is sugggested by stan org and improves sampling
  // efficiently (on uniform instead of heavy-tail)
  // - 0.1 is made for point optimization to dodge boundary case
  real<lower=0> CAUCHY_SD; // derived by MAX(RESPONSE)/constant
  real<lower=1> MIN_NU; real<lower=1> MAX_NU;

  // Seasonality Hyper-Params
  real<lower=0> SEA_SM_ALPHA;
  real<lower=0,upper=1> SEA_SM_MAX;
  int SEASONALITY;// 4 for quarterly, 12 for monthly, 52 for weekly
}
transformed data {
  int IS_SEASONAL;
  // SIGMA_EPS is a offset to dodge lower boundary case;
  real SIGMA_EPS;
  int USE_VARY_SIGMA;
  USE_VARY_SIGMA = 0;
  SIGMA_EPS = 1e-5;
  IS_SEASONAL = 0;
  if (SEASONALITY > 1) IS_SEASONAL = 1;
  # Only auto-ridge is using pr_sigma and rr_sigma
  if (REG_PENALTY_TYPE == 2) USE_VARY_SIGMA = 1;
}
parameters {
  // regression parameters
  real<lower=0> pr_sigma[NUM_OF_PR * (USE_VARY_SIGMA)];
  real<lower=0> rr_sigma[NUM_OF_RR * (USE_VARY_SIGMA)];
  // vector<lower=0,upper=BETA_MAX>[NUM_OF_PR] pr_beta;
  // vector<lower=-1 * BETA_MAX,upper=BETA_MAX>[NUM_OF_RR] rr_beta;
  vector<lower=0>[NUM_OF_PR] pr_beta;
  vector[NUM_OF_RR] rr_beta;

  real<lower=0,upper=LEV_SM_MAX> lev_sm; //level smoothing parameter
  real<lower=0,upper=SLP_SM_MAX> slp_sm; //slope smoothing parameter

  // residual tuning parameters
  // use 5*CAUCHY_SD to dodge upper boundary case
  real<lower=SIGMA_EPS,upper=5*CAUCHY_SD> obs_sigma_dummy[1 - WITH_MCMC];
  // this re-parameterization is sugggested by stan org and improves sampling
  // efficiently (on uniform instead of heavy-tail)
  // - 0.2 is made to dodge boundary case (tanh(pi/2 - 0.2) roughly equals 5 to be 
  // consistent with MAP estimation)
  real<lower=0, upper=pi()/2 - 0.2> obs_sigma_unif_dummy[WITH_MCMC];
  real<lower=MIN_NU,upper=MAX_NU> nu;

  // trend parameters
  real<lower=0,upper=1> lt_coef; // local trend coef
  real<lower=-0.5,upper=0.5> gt_coef; // global trend coef
  real<lower=0,upper=1> gt_pow; // global trend parameter

  // seasonal parameters
  //seasonality smoothing parameter
  real<lower=0,upper=SEA_SM_MAX> sea_sm[IS_SEASONAL ? 1:0];
  //initial seasonality
  vector<lower=-1,upper=1>[IS_SEASONAL ? SEASONALITY - 1:0] init_sea;
}
transformed parameters {
  real<lower=SIGMA_EPS, upper=5*CAUCHY_SD> obs_sigma;
  vector<lower=0>[NUM_OF_OBS] l;
  vector[NUM_OF_OBS] b; // slope
  vector[NUM_OF_OBS] pr; //positive regression component
  vector[NUM_OF_OBS] rr; //regular regression component
  vector[NUM_OF_OBS] r; //regression component
  vector[NUM_OF_OBS] lgt_sum; // integrated trend - sum of local & global trend
  vector[NUM_OF_OBS] yhat; // response prediction
  //seasonality vector with 1-cycle upfront as the initial condition
  vector[(NUM_OF_OBS + SEASONALITY) * IS_SEASONAL] s;

  // compute regression
  if (NUM_OF_PR > 0)
    pr = PR_MAT * pr_beta;
  else
    pr = rep_vector(0, NUM_OF_OBS);
  if (NUM_OF_RR>0)
    rr = RR_MAT * rr_beta;
  else
    rr = rep_vector(0, NUM_OF_OBS);
  r = pr + rr;

  // states initial condition
  if (IS_SEASONAL) {
    real sum_init_sea;
    sum_init_sea = 0;
    for(i in 1:(SEASONALITY - 1)){
        sum_init_sea += init_sea[i];
        s[i] = init_sea[i];
    }
    // making sure the first cycle components sum up to zero
    s[SEASONALITY] = -1 * sum_init_sea;
    s[SEASONALITY + 1] = init_sea[1];
  }
  b[1] = 0;
  if (IS_SEASONAL) {
    l[1] = RESPONSE[1] - s[1] - r[1];
  } else {
    l[1] = RESPONSE[1] - r[1];
  }
  lgt_sum[1] = l[1];
  yhat[1] = RESPONSE[1];
  for (t in 2:NUM_OF_OBS) {
    real s_t; // a transformed variable of seasonal component at time t
    if (IS_SEASONAL) {
        s_t = s[t];
    } else {
        s_t = 0.0;
    }
    
    // forecast process
    lgt_sum[t] = l[t-1] + gt_coef * fabs(l[t-1]) ^ gt_pow + lt_coef * b[t-1];
    yhat[t] = lgt_sum[t] + s_t + r[t];
    
    // update process
    // l[t] update equation with l[t-1] ONLY by excluding b[t-1];
    // It is intentionally different from the Holt-Winter form
    // The change is suggested from Slawek's original SLGT model
    l[t] = lev_sm * (RESPONSE[t] - s_t - r[t]) + (1 - lev_sm) * l[t-1];
    b[t] = slp_sm * (l[t] - l[t-1]) + (1 - slp_sm) * b[t-1];
    // with parameterization as mentioned in 7.3 "Forecasting: Principles and Practice"
    // we can safely use "l[t]" instead of "l[t-1] + b[t-1]" where 0 < sea_sm < 1
    // otherwise with original one, use 0 < sea_sm < 1 - lev_sm
    if (IS_SEASONAL)
      s[t + SEASONALITY] = sea_sm[1] * (RESPONSE[t] - l[t] - r[t]) + (1 - sea_sm[1]) * s_t;

  }
  if (WITH_MCMC) {
    // eqv. to obs_sigma ~ cauchy(SIGMA_EPS, CAUCHY_SD) T[SIGMA_EPS, ];
    obs_sigma = SIGMA_EPS + CAUCHY_SD * tan(obs_sigma_unif_dummy[1]); 
  } else {
    obs_sigma = obs_sigma_dummy[1]; 
  }
}
model {
  // prior for residuals
  if (WITH_MCMC == 0) {
    // for MAP, set finite boundary else use uniform with transformation
    obs_sigma_dummy[1] ~ cauchy(SIGMA_EPS, CAUCHY_SD) T[SIGMA_EPS, 5 * CAUCHY_SD];
  }
  for (t in 2:NUM_OF_OBS) {
    RESPONSE[t] ~ student_t(nu, yhat[t], obs_sigma);
  }
  
  // prior for seasonality
  for (i in 1:(SEASONALITY - 1))
    init_sea[i] ~ normal(0, 0.33); // 33% lift is with 1 sd prob.
    
  // regression prior
  // see these references for details
  // 1. https://jrnold.github.io/bayesian_notes/shrinkage-and-regularized-regression.html
  // 2. https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html#33_wide_weakly_informative_prior
  if (NUM_OF_PR > 0) {
    if (REG_PENALTY_TYPE== 0) {
      // fixed penalty ridge
      pr_beta ~ normal(PR_BETA_PRIOR, PR_SIGMA_PRIOR);
    } else if (REG_PENALTY_TYPE == 1) {
      // lasso penalty
      pr_beta ~ double_exponential(PR_BETA_PRIOR, LASSO_SCALE);
    } else if (REG_PENALTY_TYPE == 2) {
      // data-driven penalty for ridge
      //weak prior for sigma
      for(i in 1:NUM_OF_PR) {
        pr_sigma[i] ~ cauchy(0, AUTO_RIDGE_SCALE) T[0,];
      }
      //weak prior for betas
      pr_beta ~ normal(PR_BETA_PRIOR, pr_sigma);
    }
  }
  if (NUM_OF_RR > 0) {
    if (REG_PENALTY_TYPE == 0) {
      // fixed penalty ridge
      rr_beta ~ normal(RR_BETA_PRIOR, RR_SIGMA_PRIOR);
    } else if (REG_PENALTY_TYPE == 1) {
      // lasso penalty
      rr_beta ~ double_exponential(RR_BETA_PRIOR, LASSO_SCALE);
    } else if (REG_PENALTY_TYPE == 2) {
      // data-driven penalty for ridge
      //weak prior for sigma
      for(i in 1:NUM_OF_RR) {
        rr_sigma[i] ~ cauchy(0, AUTO_RIDGE_SCALE) T[0,];
      }
      //weak prior for betas
      rr_beta ~ normal(RR_BETA_PRIOR, rr_sigma);
    }
  }
  
  // TEST (no need to use adjusted R2 since we are purely using it as metric within a model)
  // REASON: We want to make sure our regression component is as useful as condition w/o
  // dynamic trend to avoid overfit with weird combination of r(t) and u(t) + s(t)
  if (NUM_OF_PR + NUM_OF_RR > 0) {
    // vector[NUM_OF_OBS] ybar;
    vector[NUM_OF_OBS] diff_tot;
    vector[NUM_OF_OBS] diff_res;
    real ss_tot;
    real ss_res;
    real rsq;
    real ybar;
    real new_ybar;
    
    ybar = mean(RESPONSE);
    diff_tot = RESPONSE - ybar;
    ss_tot = sum(diff_tot .* diff_tot);
    new_ybar = mean(RESPONSE - r);
    diff_res = RESPONSE - r - new_ybar;
    ss_res = sum(diff_res .* diff_res);
    rsq = 1 - ss_res/ss_tot;
    // print(rsq);
    // square-root make deminishing incentives of optimizing r square
    target += R_SQUARED_PENALTY * rsq;
  }
}
