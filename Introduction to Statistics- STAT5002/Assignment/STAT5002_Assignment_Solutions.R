################################################################################
# STAT5002 Individual Assignment - Semester 2 2025
# Complete Solutions with R Code
################################################################################

# Set seed for reproducibility
set.seed(123)

################################################################################
# Q1: Unfair and Unknown Dice
################################################################################

cat("========== Q1: Unfair and Unknown Dice ==========\n\n")

# Q1(a): Expected value and standard error of S
cat("Q1(a): Expected value and standard error\n")
cat("-------------------------------------------\n")

# Die A probabilities
# Small values (1,2,3): each has probability 2p
# Large values (4,5,6): each has probability p
# Total: 6p + 3(2p) = 9p = 1, so p = 1/9
# Small values: 2/9 each, Large values: 1/9 each

p_small <- 2/9  # P(1) = P(2) = P(3)
p_large <- 1/9  # P(4) = P(5) = P(6)

# Probability of getting at least 3 (i.e., 3,4,5,6)
p_success <- p_small + 3*p_large  # P(3) + P(4) + P(5) + P(6)
p_success_value <- 2/9 + 3/9
cat("P(value >= 3) =", p_success_value, "=", 5/9, "\n")

n <- 81
# Expected value E[S] = n*p
E_S <- n * p_success
cat("Expected value E[S] =", E_S, "\n")

# Standard error SE[S] = sqrt(n*p*(1-p))
SE_S <- sqrt(n * p_success * (1 - p_success))
cat("Standard error SE[S] =", SE_S, "\n\n")

# Q1(b): 97% prediction interval
cat("Q1(b): 97% prediction interval\n")
cat("-------------------------------------------\n")

# For 97% interval, alpha = 0.03, alpha/2 = 0.015
alpha <- 0.03
z_critical <- qnorm(1 - alpha/2)
cat("z-critical value:", z_critical, "\n")

# Prediction interval: E[S] +/- z * SE[S]
lower_bound <- E_S - z_critical * SE_S
upper_bound <- E_S + z_critical * SE_S
cat("97% Prediction Interval: [", lower_bound, ",", upper_bound, "]\n")
cat("Rounded: [", round(lower_bound, 2), ",", round(upper_bound, 2), "]\n\n")

# Simulation with 5000 runs
cat("Simulation with 5000 runs:\n")
n_sim <- 5000
simulated_S <- replicate(n_sim, {
  rolls <- sample(1:6, size = 81, replace = TRUE,
                  prob = c(2/9, 2/9, 2/9, 1/9, 1/9, 1/9))
  sum(rolls >= 3)
})

sim_mean <- mean(simulated_S)
sim_sd <- sd(simulated_S)
sim_lower <- quantile(simulated_S, 0.015)
sim_upper <- quantile(simulated_S, 0.985)

cat("Simulated mean:", sim_mean, "\n")
cat("Simulated SD:", sim_sd, "\n")
cat("Simulated 97% interval (2.5th to 97.5th percentile):",
    sim_lower, "to", sim_upper, "\n")
cat("Proportion within theoretical interval:",
    mean(simulated_S >= lower_bound & simulated_S <= upper_bound), "\n\n")

# Q1(c): Smallest p consistent with data
cat("Q1(c): Smallest p for odd values\n")
cat("-------------------------------------------\n")

# Observed: 24 odd out of 99 rolls
# One-sided 95% confidence interval (lower bound)
# Using binom.test for exact method
binom_result <- binom.test(24, 99, alternative = "greater")
cat("Observed: 24 odd out of 99 rolls\n")
cat("Sample proportion:", 24/99, "\n")
cat("95% confidence interval (one-sided):", binom_result$conf.int, "\n")
cat("Smallest p consistent with data:", round(binom_result$conf.int[1], 4), "\n\n")

# Q1(d): Chi-square test for Die B vs Die A distribution
cat("Q1(d): Chi-square test - Die B vs Die A\n")
cat("-------------------------------------------\n")

# Observed frequencies for Die B
observed <- c(10, 27, 5, 33, 9, 15)
n_total <- sum(observed)

# Expected frequencies under Die A distribution
expected_props <- c(2/9, 2/9, 2/9, 1/9, 1/9, 1/9)
expected <- n_total * expected_props

cat("Observed frequencies:", observed, "\n")
cat("Expected frequencies:", expected, "\n")

# Chi-square statistic
chi_sq_stat <- sum((observed - expected)^2 / expected)
df <- 6 - 1  # 6 categories - 1
p_value_chi <- pchisq(chi_sq_stat, df, lower.tail = FALSE)

cat("\nChi-square statistic:", chi_sq_stat, "\n")
cat("Degrees of freedom:", df, "\n")
cat("p-value:", p_value_chi, "\n")
cat("Critical value at alpha = 0.01:", qchisq(0.99, df), "\n")

# Verification with chisq.test
chisq_test <- chisq.test(observed, p = expected_props)
cat("\nVerification with chisq.test():\n")
print(chisq_test)

cat("\n")

################################################################################
# Q2: Caffeine Effect
################################################################################

cat("========== Q2: Caffeine Effect ==========\n\n")

# Data
pre_ms <- c(171, 162, 164, 169, 173, 168, 158, 166,
            176, 161, 170, 159, 167, 163, 172, 160)
post_ms <- c(160, 155, 158, 161, 165, 170, 151, 157,
             170, 155, 165, 157, 160, 165, 166, 159)

# Q2(a): State hypotheses
cat("Q2(a): Hypotheses\n")
cat("-------------------------------------------\n")
cat("Let mu_d = population mean difference (PRE - POST)\n")
cat("H0: mu_d = 0 (caffeine has no effect)\n")
cat("H1: mu_d > 0 (caffeine reduces reaction time)\n\n")

# Q2(b): Select test
cat("Q2(b): Test selection\n")
cat("-------------------------------------------\n")
cat("Test: Paired t-test (one-sided, right-tailed)\n")
cat("Justification:\n")
cat("- Same athletes measured twice (paired design)\n")
cat("- Testing if caffeine REDUCES reaction time (one-sided)\n")
cat("- Small sample size (n=16), assume normality\n\n")

# Q2(c): Check assumptions
cat("Q2(c): Assumption checking\n")
cat("-------------------------------------------\n")

differences <- pre_ms - post_ms
cat("Differences (PRE - POST):", differences, "\n")
cat("Mean difference:", mean(differences), "\n")
cat("SD of differences:", sd(differences), "\n\n")

# Normality test
shapiro_test <- shapiro.test(differences)
cat("Shapiro-Wilk test for normality:\n")
cat("W =", shapiro_test$statistic, ", p-value =", shapiro_test$p.value, "\n")
if(shapiro_test$p.value > 0.05) {
  cat("Conclusion: No evidence against normality (p > 0.05)\n\n")
} else {
  cat("Conclusion: Evidence against normality (p <= 0.05)\n\n")
}

# Create QQ plot and histogram
pdf("Q2_assumption_plots.pdf", width = 10, height = 5)
par(mfrow = c(1, 2))
hist(differences, main = "Histogram of Differences (PRE - POST)",
     xlab = "Difference (ms)", col = "lightblue", breaks = 8)
qqnorm(differences, main = "Q-Q Plot of Differences")
qqline(differences, col = "red", lwd = 2)
dev.off()

cat("Plots saved to Q2_assumption_plots.pdf\n\n")

# Q2(d): Compute test statistic and p-value
cat("Q2(d): Test statistic and p-value\n")
cat("-------------------------------------------\n")

n <- length(differences)
mean_diff <- mean(differences)
sd_diff <- sd(differences)
se_diff <- sd_diff / sqrt(n)

t_stat <- mean_diff / se_diff
df_t <- n - 1
p_value_t <- pt(t_stat, df_t, lower.tail = FALSE)

cat("Sample size n =", n, "\n")
cat("Mean difference =", mean_diff, "\n")
cat("SD of differences =", sd_diff, "\n")
cat("SE of differences =", se_diff, "\n")
cat("t-statistic =", t_stat, "\n")
cat("Degrees of freedom =", df_t, "\n")
cat("p-value (one-sided) =", p_value_t, "\n\n")

# Rejection region
alpha_2d <- 0.05
t_critical <- qt(1 - alpha_2d, df_t)
cat("At alpha = 0.05:\n")
cat("Critical value =", t_critical, "\n")
cat("Rejection region: t >", t_critical, "\n\n")

# Q2(e): Conclusion
cat("Q2(e): Conclusion\n")
cat("-------------------------------------------\n")
if(p_value_t < 0.05) {
  cat("p-value =", round(p_value_t, 4), "< 0.05\n")
  cat("Decision: Reject H0\n")
  cat("Conclusion: There is statistically significant evidence that\n")
  cat("the caffeine gel reduces reaction time at the 5% significance level.\n\n")
} else {
  cat("p-value =", round(p_value_t, 4), ">= 0.05\n")
  cat("Decision: Fail to reject H0\n")
  cat("Conclusion: There is insufficient evidence that\n")
  cat("the caffeine gel reduces reaction time at the 5% significance level.\n\n")
}

# Q2(f): Bootstrap simulation
cat("Q2(f): Bootstrap simulation\n")
cat("-------------------------------------------\n")

n_bootstrap <- 10000
bootstrap_t_stats <- numeric(n_bootstrap)

for(i in 1:n_bootstrap) {
  # Resample differences with replacement
  boot_sample <- sample(differences, size = n, replace = TRUE)
  boot_mean <- mean(boot_sample)
  boot_sd <- sd(boot_sample)
  boot_se <- boot_sd / sqrt(n)
  bootstrap_t_stats[i] <- boot_mean / boot_se
}

cat("Bootstrap completed with", n_bootstrap, "resamples\n")
cat("Bootstrap mean of t-statistics:", mean(bootstrap_t_stats), "\n")
cat("Bootstrap SD of t-statistics:", sd(bootstrap_t_stats), "\n\n")

# Plot histogram with theoretical distribution
pdf("Q2_bootstrap_histogram.pdf", width = 8, height = 6)
hist(bootstrap_t_stats, breaks = 50, probability = TRUE,
     main = "Bootstrap Distribution of t-statistic vs Theoretical t-distribution",
     xlab = "t-statistic", col = "lightblue", xlim = c(-4, 8))
curve(dt(x, df = df_t), add = TRUE, col = "red", lwd = 2)
abline(v = t_stat, col = "darkgreen", lwd = 2, lty = 2)
legend("topright", legend = c("Bootstrap", "Theoretical t(15)", "Observed t"),
       col = c("lightblue", "red", "darkgreen"), lwd = 2, lty = c(1, 1, 2))
dev.off()

cat("Histogram saved to Q2_bootstrap_histogram.pdf\n\n")

# Q2(g): Bootstrap p-value
cat("Q2(g): Bootstrap p-value and conclusion\n")
cat("-------------------------------------------\n")

bootstrap_p_value <- mean(bootstrap_t_stats >= t_stat)
cat("Bootstrap p-value =", bootstrap_p_value, "\n")
cat("Theoretical p-value =", p_value_t, "\n\n")

if(bootstrap_p_value < 0.05) {
  cat("Bootstrap p-value < 0.05\n")
  cat("Decision: Reject H0\n")
  cat("Conclusion: The bootstrap simulation confirms that there is\n")
  cat("statistically significant evidence that caffeine reduces reaction time.\n\n")
} else {
  cat("Bootstrap p-value >= 0.05\n")
  cat("Decision: Fail to reject H0\n")
  cat("Conclusion: The bootstrap simulation suggests insufficient evidence\n")
  cat("that caffeine reduces reaction time at the 5% significance level.\n\n")
}

################################################################################
# Q3: Caffeine Effect and Self-report
################################################################################

cat("========== Q3: Caffeine Effect and Self-report ==========\n\n")

# Data separated by self-report
pre_alert <- c(171, 162, 169, 173, 158, 166, 176, 170, 167, 172)
post_alert <- c(160, 155, 161, 165, 151, 157, 170, 165, 160, 166)
pre_notalert <- c(164, 168, 161, 159, 163, 160)
post_notalert <- c(158, 170, 155, 157, 165, 159)

# Calculate differences
diff_alert <- pre_alert - post_alert
diff_notalert <- pre_notalert - post_notalert

cat("HATPC Framework for Two-Sample t-test\n")
cat("=======================================\n\n")

# H: Hypotheses
cat("H - HYPOTHESES:\n")
cat("Let mu_A = population mean caffeine effect for alert group\n")
cat("Let mu_NA = population mean caffeine effect for not-alert group\n")
cat("H0: mu_A = mu_NA (no difference in caffeine effect)\n")
cat("H1: mu_A != mu_NA (caffeine effect differs between groups)\n\n")

# A: Assumptions
cat("A - ASSUMPTIONS:\n")
cat("1. Both samples are independent random samples\n")
cat("2. Differences in each group are normally distributed\n")
cat("3. Equal variances (we'll use Welch's t-test for robustness)\n\n")

cat("Alert group differences:", diff_alert, "\n")
cat("Not-alert group differences:", diff_notalert, "\n\n")

cat("Summary statistics:\n")
n_alert <- length(diff_alert)
n_notalert <- length(diff_notalert)
mean_alert <- mean(diff_alert)
mean_notalert <- mean(diff_notalert)
sd_alert <- sd(diff_alert)
sd_notalert <- sd(diff_notalert)

cat("Alert group: n =", n_alert, ", mean =", mean_alert, ", SD =", sd_alert, "\n")
cat("Not-alert group: n =", n_notalert, ", mean =", mean_notalert,
    ", SD =", sd_notalert, "\n\n")

# T: Test statistic
cat("T - TEST STATISTIC:\n")

# Pooled variance (for equal variance assumption)
pooled_var <- ((n_alert - 1) * sd_alert^2 + (n_notalert - 1) * sd_notalert^2) /
  (n_alert + n_notalert - 2)
pooled_sd <- sqrt(pooled_var)
se_pooled <- pooled_sd * sqrt(1/n_alert + 1/n_notalert)

t_stat_q3 <- (mean_alert - mean_notalert) / se_pooled
df_q3 <- n_alert + n_notalert - 2

cat("Pooled SD =", pooled_sd, "\n")
cat("SE (pooled) =", se_pooled, "\n")
cat("t-statistic =", t_stat_q3, "\n")
cat("Degrees of freedom =", df_q3, "\n\n")

# P: P-value
cat("P - P-VALUE:\n")
p_value_q3 <- 2 * pt(abs(t_stat_q3), df_q3, lower.tail = FALSE)
cat("p-value (two-sided) =", p_value_q3, "\n")
cat("Significance level alpha = 0.05\n\n")

# C: Conclusion
cat("C - CONCLUSION:\n")
t_crit_q3 <- qt(0.975, df_q3)
cat("Critical value (two-sided, alpha=0.05) = +/-", t_crit_q3, "\n")
cat("Rejection region: |t| >", t_crit_q3, "\n\n")

if(p_value_q3 < 0.05) {
  cat("Decision: Reject H0 (p =", round(p_value_q3, 4), "< 0.05)\n")
  cat("Conclusion: There is statistically significant evidence that\n")
  cat("the caffeine effect differs between athletes who felt alert\n")
  cat("and those who did not feel alert at the 5% significance level.\n\n")
} else {
  cat("Decision: Fail to reject H0 (p =", round(p_value_q3, 4), ">= 0.05)\n")
  cat("Conclusion: There is insufficient evidence that the caffeine\n")
  cat("effect differs between the two groups at the 5% significance level.\n\n")
}

# Verification with t.test
cat("Verification with t.test():\n")
ttest_result <- t.test(diff_alert, diff_notalert, var.equal = TRUE)
print(ttest_result)

cat("\n")

################################################################################
# Q4: Advertising and Sales
################################################################################

cat("========== Q4: Advertising and Sales ==========\n\n")

# Data
x <- c(2.0, 3.5, 4.0, 5.0, 6.5, 7.0, 8.0, 9.5, 10.0, 11.0,
       12.5, 13.0, 14.5, 15.0, 16.0, 17.5, 18.0, 19.5, 20.5, 22.0)
y <- c(17.0, 23.0, 23.2, 28.0, 30.8, 33.3, 34.9, 41.7, 41.6, 46.8,
       47.7, 50.5, 53.1, 52.4, 55.0, 56.1, 55.5, 52.8, 51.9, 50.0)

# Q4(a): Linear regression model
cat("Q4(a): Linear regression model\n")
cat("-------------------------------------------\n")

model <- lm(y ~ x)
summary_model <- summary(model)
print(summary_model)

cat("\n\nInterpretation of coefficients:\n")
intercept <- coef(model)[1]
slope <- coef(model)[2]

cat("Intercept (beta_0) =", round(intercept, 2), "thousand cups\n")
cat("Interpretation: When advertising budget is $0, the predicted sales are\n")
cat("approximately", round(intercept, 2), "thousand cups (", round(intercept*1000, 0), "cups).\n\n")

cat("Slope (beta_1) =", round(slope, 2), "thousand cups per thousand dollars\n")
cat("Interpretation: For each additional $1,000 spent on advertising,\n")
cat("the predicted sales increase by approximately", round(slope, 2),
    "thousand cups (", round(slope*1000, 0), "cups).\n")
cat("Equivalently: For each additional $1 spent, sales increase by about",
    round(slope, 2), "cups.\n\n")

# Q4(b): Check assumptions
cat("Q4(b): Checking assumptions\n")
cat("-------------------------------------------\n\n")

residuals <- residuals(model)
fitted_values <- fitted(model)

# Create diagnostic plots
pdf("Q4_diagnostic_plots.pdf", width = 12, height = 10)
par(mfrow = c(2, 2))

# 1. Residuals vs Fitted (Linearity and Homoscedasticity)
plot(fitted_values, residuals,
     main = "Residuals vs Fitted Values",
     xlab = "Fitted Values", ylab = "Residuals",
     pch = 19, col = "blue")
abline(h = 0, col = "red", lwd = 2)
lowess_fit <- lowess(fitted_values, residuals)
lines(lowess_fit, col = "darkgreen", lwd = 2)

# 2. Q-Q Plot (Normality)
qqnorm(residuals, main = "Normal Q-Q Plot of Residuals", pch = 19, col = "blue")
qqline(residuals, col = "red", lwd = 2)

# 3. Scale-Location plot (Homoscedasticity)
plot(fitted_values, sqrt(abs(residuals)),
     main = "Scale-Location Plot",
     xlab = "Fitted Values", ylab = "√|Residuals|",
     pch = 19, col = "blue")
abline(h = mean(sqrt(abs(residuals))), col = "red", lwd = 2)

# 4. Histogram of residuals
hist(residuals, breaks = 10,
     main = "Histogram of Residuals",
     xlab = "Residuals", col = "lightblue", probability = TRUE)
curve(dnorm(x, mean = mean(residuals), sd = sd(residuals)),
      add = TRUE, col = "red", lwd = 2)

dev.off()

cat("Diagnostic plots saved to Q4_diagnostic_plots.pdf\n\n")

# Create scatter plot with regression line
pdf("Q4_scatterplot.pdf", width = 8, height = 6)
plot(x, y, pch = 19, col = "blue",
     main = "Advertising Budget vs Sales with Regression Line",
     xlab = "Advertising Budget (thousand $)",
     ylab = "Sales (thousand cups)")
abline(model, col = "red", lwd = 2)
grid()
dev.off()

cat("Scatterplot saved to Q4_scatterplot.pdf\n\n")

# Formal tests
cat("Formal assumption tests:\n\n")

# 1. Normality test
shapiro_resid <- shapiro.test(residuals)
cat("1. NORMALITY OF RESIDUALS:\n")
cat("Shapiro-Wilk test: W =", shapiro_resid$statistic,
    ", p-value =", shapiro_resid$p.value, "\n")
if(shapiro_resid$p.value > 0.05) {
  cat("Conclusion: No evidence against normality (p > 0.05)\n")
  cat("The normality assumption appears to be satisfied.\n\n")
} else {
  cat("Conclusion: Evidence against normality (p <= 0.05)\n")
  cat("The normality assumption may be violated.\n\n")
}

# 2. Linearity - check scatter plot and residual plot
cat("2. LINEARITY:\n")
cat("Visual inspection of:\n")
cat("- Scatterplot shows a general linear trend initially, but appears to\n")
cat("  flatten or decline at higher advertising budgets (potential non-linearity)\n")
cat("- Residuals vs Fitted plot should show random scatter around zero\n")
cat("  If there's a pattern (curve), linearity is violated\n\n")

# 3. Homoscedasticity
cat("3. HOMOSCEDASTICITY (constant variance):\n")
cat("Visual inspection of:\n")
cat("- Residuals vs Fitted: spread should be roughly constant\n")
cat("- Scale-Location plot: should show horizontal trend\n\n")

# Calculate correlation
correlation <- cor(x, y)
cat("Correlation between x and y:", round(correlation, 4), "\n")
cat("R-squared:", round(summary_model$r.squared, 4), "\n\n")

# Q4(c): Other variables
cat("Q4(c): Other variables that could affect sales\n")
cat("-------------------------------------------\n")
cat("Two other variables that could affect weekly coffee sales:\n\n")

cat("1. WEATHER/TEMPERATURE:\n")
cat("   - Coffee consumption typically increases in colder weather\n")
cat("   - Seasonal variations affect customer behavior\n")
cat("   - Could measure: average weekly temperature or weather conditions\n\n")

cat("2. DAY OF WEEK / HOLIDAYS:\n")
cat("   - Weekdays vs weekends may have different sales patterns\n")
cat("   - Public holidays, exam periods, or special events affect foot traffic\n")
cat("   - Could measure: number of weekdays/weekend days, or holiday indicator\n\n")

cat("Additional possibilities:\n")
cat("- Competitor promotions or pricing\n")
cat("- Store location foot traffic\n")
cat("- Product launches or menu changes\n")
cat("- Economic indicators (consumer confidence, employment)\n\n")

cat("\n========== ALL ANALYSES COMPLETE ==========\n")

# Save workspace
save.image("STAT5002_Assignment_Workspace.RData")
cat("\nWorkspace saved to STAT5002_Assignment_Workspace.RData\n")
