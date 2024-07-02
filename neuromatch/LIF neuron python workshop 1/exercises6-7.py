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

# Exercise 6

print(t_max, dt, tau, el, vr, vth, r, i_mean)
# Initialize step_end and v0


step_end = int(t_max / dt)
v = el

plt.figure()
plt.title('$V_m$ with sinusoidal I(t)')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)')

# Loop for step_end steps
for step in range(step_end):
  # Compute value of t
  t = step * dt
  
  # Compute value of i at this time step
  i = i_mean * (1 + np.sin((t * 2 * np.pi) / 0.01))

  # Compute v
  v = v + dt/tau*(el - v + r*i)

  plt.plot(t, v, 'k.')

  # Print value of t and v
  print(f"{t:.3f} {v:.4e}")

plt.show()

# Exercise 7

np.random.seed(2020)


step_end = int(t_max / dt)
v = el

plt.figure()
plt.title('$V_m$ with random I(t)')
plt.xlabel('time (s)')
plt.ylabel('$V_m$ (V)')




# Loop for step_end steps
for step in range(step_end):
  # Compute value of t
  t = step * dt
  
  random_num = np.random.random()*2-1

  # Compute value of i at this time step
  i = i_mean * (1+0.1*((t_max/dt)**0.5)*random_num)

  # Compute v
  v = v + dt/tau*(el - v + r*i)

  plt.plot(t, v, 'k.')

  # Print value of t and v
  print(f"{t:.3f} {v:.4e}")

plt.show()

