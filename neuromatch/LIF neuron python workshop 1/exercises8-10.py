import numpy as np
import matplotlib.pyplot as plt

t_max = 150e-3   # second
dt = 1e-3        # second
tau = 20e-3      # second
el = -60e-3      # milivolt
vr = -70e-3      # milivolt
vth = -50e-3     # milivolt
r = 100e6        # ohm
i_mean = 25e-11  # ampere


# Set random number generator
np.random.seed(2020)

# Initialize step_end and n
step_end = int(t_max / dt)
n = 50

# Intiatialize the list v_n with 50 values of membrane leak potential el
v_n = [el] * n

with plt.xkcd():
  # Initialize the figure
  plt.figure()
  plt.title('Multiple realizations of $V_m$')
  plt.xlabel('time (s)')
  plt.ylabel('$V_m$ (V)')

  # Loop for step_end steps
  for step in range(step_end):

    # Compute value of t
    t = step * dt

    # Loop for n simulations
    for j in range(0, n):

      # Compute value of i at this time step
      i = i_mean * (1 + 0.1 * (t_max/dt)**(0.5) * (2* np.random.random() - 1))

      # Compute value of v for this simulation
      v_n[j] = v_n[j] + dt/tau * (el - v_n[j] + r*i)


    v_mean = np.sum(v_n)/n

    v_var_n = [(v - v_mean)**2 for v in v_n]

    # Compute sample variance v_var by summing the values of v_var_n with sum and dividing by n-1
    v_var = 1/(n-1) * np.sum(v_var_n)

    # Compute the standard deviation v_std with the function np.sqrt
    v_std = np.sqrt(v_var)


    # Plot all simulations (use alpha = 0.1 to make each marker slightly transparent)
    plt.plot([t]*n, v_n, '.k', alpha = 0.1)

    plt.plot(t, v_mean, 'C0.', alpha = 0.8)

    plt.plot(t, v_mean + v_std, 'C7.', alpha =0.8)

    # Plot mean - standard deviation with alpha=0.8 and argument 'C7.'
    plt.plot(t, v_mean - v_std, 'C7.', alpha =0.8)



  # Display plot
  plt.show()
