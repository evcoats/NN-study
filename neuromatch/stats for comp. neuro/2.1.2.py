from helpers import *

# Transition matrix
transition_matrix = np.array([[ 0.2, 0.6, 0.2], [ .6, 0.3, 0.1], [0.8, 0.2, 0]])

# Initial state, p0
p0 = np.array([0, 1, 0])

# Compute the probabilities 4 transitions later (use np.linalg.matrix_power to raise a matrix a power)
p4 = p0 @ np.linalg.matrix_power(transition_matrix,4)

# The second area is indexed as 1 (Python starts indexing at 0)
print(f"The probability the rat will be in area 2 after 4 transitions is: {p4[1]}")


# Initialize random initial distribution
p_random = np.ones((1, 3))/3

# Fill in the missing line to get the state matrix after 100 transitions, like above
p_average_time_spent = p_random @ np.linalg.matrix_power(transition_matrix, 100)


print(f"The proportion of time spend by the rat in each of the three states is: {p_average_time_spent[0]}")
