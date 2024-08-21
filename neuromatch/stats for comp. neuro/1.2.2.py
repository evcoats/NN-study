from helpers import * 

# Set random seed
np.random.seed(0)

# Draw 5 samples from a Poisson distribution with lambda = 4
sampled_spike_counts = np.random.poisson(lam=4,size=5)

# Print the counts
print("The samples drawn from the Poisson distribution are " +
          str(sampled_spike_counts))
