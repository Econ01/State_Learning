import numpy as np
from qutip import wigner, Qobj, wigner_cmap
from thewalrus.quantum import state_vector, density_matrix

import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import operation
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from strawberryfields.ops import *
from mpl_toolkits.mplot3d import Axes3D
import math

# Cutoff dimension
cutoff = 10

# Number of layers
depth = 15

# Number of steps in optimization routine performing gradient descent
reps = 100

# Learning rate
lr = 0.05

# Standard deviation of initial parameters
passive_sd = 0.1
active_sd = 0.001

import tensorflow as tf

# set the random seed
tf.random.set_seed(42)

# squeeze gate
sq_r = tf.random.normal(shape=[depth], stddev=active_sd)
sq_phi = tf.random.normal(shape=[depth], stddev=passive_sd)

# displacement gate
d_r = tf.random.normal(shape=[depth], stddev=active_sd)
d_phi = tf.random.normal(shape=[depth], stddev=passive_sd)

# rotation gates
r1 = tf.random.normal(shape=[depth], stddev=passive_sd)
r2 = tf.random.normal(shape=[depth], stddev=passive_sd)

# kerr gate
kappa = tf.random.normal(shape=[depth], stddev=active_sd)

weights = tf.convert_to_tensor([r1, sq_r, sq_phi, r2, d_r, d_phi, kappa])
weights = tf.Variable(tf.transpose(weights))

# Single-mode Strawberry Fields program
prog = sf.Program(1)
prog_2 = sf.Program(1)

# Create the 7 Strawberry Fields free parameters for each layer
sf_params = []
names = ["r1", "sq_r", "sq_phi", "r2", "d_r", "d_phi", "kappa"]

for i in range(depth):
    # For the ith layer, generate parameter names "r1_i", "sq_r_i", etc.
    sf_params_names = ["{}_{}".format(n, i) for n in names]
    # Create the parameters, and append them to our list ``sf_params``.
    sf_params.append(prog.params(*sf_params_names))
    sf_params.append(prog_2.params(*sf_params_names))

sf_params = np.array(sf_params)
print(sf_params.shape)

# layer architecture
@operation(1)
def layer(i, q):
    Rgate(sf_params[i][0]) | q
    Sgate(sf_params[i][1], sf_params[i][2]) | q
    Rgate(sf_params[i][3]) | q
    Dgate(sf_params[i][4], sf_params[i][5]) | q
    Kgate(sf_params[i][6]) | q
    return q

# Apply circuit of layers with corresponding depth
with prog.context as q:
    for k in range(depth):
        layer(k) | q[0]

with prog_2.context as q:
    for k in range(depth):
        layer(k) | q[0]

eng = sf.Engine("tf", backend_options={"cutoff_dim": cutoff})

import numpy as np

#----------------------------------------------------------------------------------------

# First target state
target_state = np.zeros([cutoff])
target_state[1] = 1
print(target_state)

# Second target state
prog_1 = sf.Program(1)
eng_1 = sf.Engine("gaussian")

with prog_1.context as q:
    sf.ops.Squeezed(r=1, p=0) | q

state_1 = eng_1.run(prog_1).state
mu, cov = state_1.means(), state_1.cov()

squeezed = state_vector(mu, cov ,normalize=False, cutoff=cutoff)
squeezed_psi = np.linalg.norm(squeezed)
squeezed = squeezed / squeezed_psi

target_state_2 = squeezed


#----------------------------------------------------------------------------------------

def cost(weights):
    # Create a dictionary mapping from the names of the Strawberry Fields
    # free parameters to the TensorFlow weight values.
    mapping = {p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))}
    mapping_2 = {p.name: w for p, w in zip(sf_params.flatten(), tf.reshape(weights, [-1]))}

    # Run engine
    state = eng.run(prog, args=mapping).state
    state_2 = eng.run(prog_2, args=mapping_2).state

    # Extract the statevector
    ket = state.ket()
    ket_2 = state_2.ket()

    # Compute the fidelity between the output statevector
    # and the target state.

    fidelity = (1/3)*tf.abs(tf.reduce_sum((tf.math.conj(ket) * target_state))) ** 2 + (1/3)*tf.abs(tf.reduce_sum((tf.math.conj(ket_2) * target_state_2))) ** 2 + (1/3)*tf.abs(tf.reduce_sum((tf.math.conj(ket_2) * tf.math.conj(ket)))) ** 2

    # Objective function to minimize
    #cost = tf.abs(tf.reduce_sum(tf.math.conj(ket) * target_state) - 1)
    cost = tf.abs((1/3)*(tf.reduce_sum(tf.math.conj(ket) * target_state)+(1/3)*(tf.reduce_sum(tf.math.conj(ket_2) * target_state_2))+(1/3)*(tf.reduce_sum(tf.math.conj(ket_2) * tf.math.conj(ket)))) - 1)

    return cost, fidelity, ket, ket_2

opt = tf.keras.optimizers.Adam(learning_rate=lr)

fid_progress = []
best_fid = 0

for i in range(reps):
    # reset the engine if it has already been executed
    if eng.run_progs:
        eng.reset()

    with tf.GradientTape() as tape:
        loss, fid, ket, ket_2 = cost(weights)

    # Stores fidelity at each step
    fid_progress.append(fid.numpy())

    if fid > best_fid:
        # store the new best fidelity and best state
        best_fid = fid.numpy()
        learnt_state = ket.numpy()
        learnt_state_2 = ket_2.numpy()

    # one repetition of the optimization
    gradients = tape.gradient(loss, weights)
    opt.apply_gradients(zip([gradients], [weights]))

    # Prints progress at every rep
    if i % 1 == 0:
        print("Rep: {} Cost: {:.4f} Fidelity: {:.4f}".format(i, loss, fid))

from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.sans-serif"] = ["Computer Modern Roman"]
plt.style.use("default")

plt.plot(fid_progress)
plt.ylabel("Fidelity")
plt.xlabel("Step")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def wigner(rho):
    import copy

    # Domain parameter for Wigner function plots
    l = 5.0
    cutoff = rho.shape[0]

    # Creates 2D grid for Wigner function plots
    x = np.linspace(-l, l, 100)
    p = np.linspace(-l, l, 100)

    Q, P = np.meshgrid(x, p)
    A = (Q + P * 1.0j) / (2 * np.sqrt(2 / 2))

    Wlist = np.array([np.zeros(np.shape(A), dtype=complex) for k in range(cutoff)])

    # Wigner function for |0><0|
    Wlist[0] = np.exp(-2.0 * np.abs(A) ** 2) / np.pi

    # W = rho(0,0)W(|0><0|)
    W = np.real(rho[0, 0]) * np.real(Wlist[0])

    for n in range(1, cutoff):
        Wlist[n] = (2.0 * A * Wlist[n - 1]) / np.sqrt(n)
        W += 2 * np.real(rho[0, n] * Wlist[n])

    for m in range(1, cutoff):
        temp = copy.copy(Wlist[m])
        # Wlist[m] = Wigner function for |m><m|
        Wlist[m] = (2 * np.conj(A) * temp - np.sqrt(m) * Wlist[m - 1]) / np.sqrt(m)

        # W += rho(m,m)W(|m><m|)
        W += np.real(rho[m, m] * Wlist[m])

        for n in range(m + 1, cutoff):
            temp2 = (2 * A * Wlist[n - 1] - np.sqrt(m) * temp) / np.sqrt(n)
            temp = copy.copy(Wlist[n])
            # Wlist[n] = Wigner function for |m><n|
            Wlist[n] = temp2

            # W += rho(m,n)W(|m><n|) + rho(n,m)W(|n><m|)
            W += 2 * np.real(rho[m, n] * Wlist[n])

    return Q, P, W / 2

rho_target = np.outer(target_state, target_state.conj())
rho_learnt = np.outer(learnt_state, learnt_state.conj())

rho_target_2 = np.outer(target_state_2, target_state_2.conj())
rho_learnt_2 = np.outer(learnt_state_2, learnt_state_2.conj())

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X, P, W = wigner(rho_target)
ax.plot_surface(X, P, W, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
ax.contour(X, P, W, 10, cmap="RdYlGn", linestyles="solid", offset=-0.17)
ax.set_axis_off()
plt.title("First Target State")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X, P, W = wigner(rho_learnt)
ax.plot_surface(X, P, W, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
ax.contour(X, P, W, 10, cmap="RdYlGn", linestyles="solid", offset=-0.17)
ax.set_axis_off()
plt.title("First Trained State")
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111, projection="3d")
X1, P1, W1 = wigner(rho_target_2)
ax1.plot_surface(X1, P1, W1, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
ax1.contour(X1, P1, W1, 10, cmap="RdYlGn", linestyles="solid", offset=-0.17)
ax1.set_axis_off()
plt.title("Second Target State")
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111, projection="3d")
X1, P1, W1 = wigner(rho_learnt_2)
ax1.plot_surface(X1, P1, W1, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
ax1.contour(X1, P1, W1, 10, cmap="RdYlGn", linestyles="solid", offset=-0.17)
ax1.set_axis_off()
plt.title("Second Trained State")
plt.show()