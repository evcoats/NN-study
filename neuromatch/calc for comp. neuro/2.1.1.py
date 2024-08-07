from helpers import *
# Define time, time constant
t = np.arange(0, 10, .1)
tau = 0.5

# Compute alpha function
f = t * np.exp(-t/tau)

# Define u(t), v(t)
u_t = t
v_t = np.exp(-t/tau)

# Define du/dt, dv/dt
du_dt = 1
dv_dt = -1/tau*np.exp(-t/tau)

# Define full derivative
df_dt = dv_dt * u_t + v_t * du_dt

# Visualize
plot_alpha_func(t, f, df_dt)

plt.show()