
# Normal Distribution

The normal distribution, also known as the Gaussian distribution, is a probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. It has numerous applications in real life, including in the fields of statistics, finance, natural and social sciences.


The probability density function (PDF) of the normal distribution is given by the formula:

![eqauations](https://latex.codecogs.com/svg.image?&space;f(x|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}})

where:
- ![eqauations](https://latex.codecogs.com/svg.image?x) is the variable,
- ![eqauations](https://latex.codecogs.com/svg.image?\mu) is the mean,
- ![eqauations](https://latex.codecogs.com/svg.image?\sigma^2) is the variance,
- ![eqauations](https://latex.codecogs.com/svg.image?\sigma) is the standard deviation, and
- ![eqauations](https://latex.codecogs.com/svg.image?e) is the base of the natural logarithm.

Let's calculate the probability of a random variable ![eqauations](https://latex.codecogs.com/svg.image?X), which follows a normal distribution with a mean ![eqauations](https://latex.codecogs.com/svg.image?\mu) = ![eqauations](https://latex.codecogs.com/svg.image?50) and a standard deviation ![eqauations](https://latex.codecogs.com/svg.image?\sigma) = ![eqauations](https://latex.codecogs.com/svg.image?10), taking on a value less than ![eqauations](https://latex.codecogs.com/svg.image?60).

1. **Identify Parameters**: ![eqauations](https://latex.codecogs.com/svg.image?\mu) = ![eqauations](https://latex.codecogs.com/svg.image?50), ![eqauations](https://latex.codecogs.com/svg.image?\sigma) = ![eqauations](https://latex.codecogs.com/svg.image?10)
2. **Standardize the Variable** (convert ![eqauations](https://latex.codecogs.com/svg.image?X) to ![eqauations](https://latex.codecogs.com/svg.image?Z)-![eqauations](https://latex.codecogs.com/svg.image?score):</br>
	![equations](https://latex.codecogs.com/svg.image?&space;Z=\frac{X-\mu}{\sigma}=\frac{60-50}{10}=1&space;)
3. **Use Z-table** or cumulative distribution function (CDF) to find the probability:
   - For simplicity, let's say ![equations](https://latex.codecogs.com/svg.image?P(X<60)=P(Z<1)) corresponds to approximately ![equations](https://latex.codecogs.com/svg.image?0.8413) from the Z-table.

### 4. Python Code for Visualization


```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
mu = 50
sigma = 10
x = 60

# Plot
x_values = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
y_values = stats.norm.pdf(x_values, mu, sigma)
plt.plot(x_values, y_values, label='Normal Distribution')
plt.fill_between(x_values, y_values, where=(x_values < x), color='skyblue', alpha=0.5, label='Area under curve')
plt.legend()
plt.title('Normal Distribution ($\mu=50$, $\sigma=10$)')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.show()

# Probability Calculation
prob = stats.norm.cdf(x, mu, sigma)
print(f'Probability (X < {x}): {prob}')
```

### 5. R Code for Visualization

Now, let's create a similar visualization in R to show the normal distribution and calculate the probability.

```r
# Parameters
mu <- 50
sigma <- 10
x <- 60

# Plot
x_values <- seq(mu - 4*sigma, mu + 4*sigma, length.out = 1000)
y_values <- dnorm(x_values, mean = mu, sd = sigma)
plot(x_values, y_values, type = 'l', main = 'Normal Distribution (mu=50, sigma=10)', xlab = 'X', ylab = 'Density')
polygon(c(x_values[x_values<x], x), c(y_values[x_values<x], 0), col = 'skyblue')

# Probability Calculation
prob <- pnorm(x, mean = mu, sd = sigma)
cat(sprintf('Probability (X < %d): %f', x, prob))
```

Both Python and R code snippets above will visualize the normal distribution curve for a mean (\(\mu\)) of 50 and a standard deviation (\(\sigma\)) of 10. They also shade the area under the curve for values less than 60, which corresponds to the probability calculation for \(X < 60\). The area under the curve to the left of \(X = 60\) represents the probability we calculated in our step-by-step example.
