import numpy as np
from numba import njit, float64, int64

np.random.seed(0)

# create hashing table
HASH_TABLE_SIZE = 256
HASH_TABLE = np.arange(HASH_TABLE_SIZE)
np.random.shuffle(HASH_TABLE)
HASH_TABLE = np.concatenate((HASH_TABLE, HASH_TABLE))

# 1d gradients
GRADIENTS = np.random.choice([-1.0, 1.0], HASH_TABLE_SIZE)

# 2d gradients
rand_deg = np.random.uniform(0, 2 * np.pi, HASH_TABLE_SIZE)
rand_x = np.cos(rand_deg)
rand_y = np.sin(rand_deg)
GRADIENTS_2D = np.array([rand_x, rand_y]).T


def fractal(p,
            frequency=1.0,
            amplitude=1.0,
            n_octaves=4,
            persistence=0.5,
            lucanarity=2):
  """Sample fractal noise in one or two dimensions.

  Args:
      p (1d or 2d array): point(s) in space at which to sample fractal noise. If 
        1d, shape is (n_points). If 2d, shape is (2, n_points) where the first
        axis holds the location of each point on a 2d grid.
      frequency (float, list of floats): either the frequency of the first
        octave or a list of frequencies for each octave. Defaults to 1.0
      amplitude (float, optional): either the amplitude of the first
        octave or a list of amplitude for each octave. Defaults to 1.0.
      n_octaves (int, optional): number of noise octaves to add. Defaults to 4.
      persistence (float, optional): amount the amplitude of subsequent
        octaves gets scaled. Ignored if 'amplitude' is a list. Defaults to 0.5.
      lucanarity (int, optional): amount the frequency of subsequent
        octaves gets scaled. Ignored if 'frequency' is a list.. Defaults to 2.

  Returns:
      1d np array: sampled noise at each point

  Example 1d usage:
    n_p, o_p, d_p = 1000, 0.0, 1
    p = np.linspace(o_p, o_p + n_p * d_p, n_p, endpoint=False)
    fractal_noise = fractal(p)

  Example 2d usage:
    nx, ox, dx = 1000, 0.0, 1
    ny, oy, dy = 1000, 0.0, 1
    x = np.linspace(ox, ox + nx * dx, nx, endpoint=False)
    y = np.linspace(oy, oy + ny * dy, ny, endpoint=False)
    p = np.stack(np.meshgrid(x, y, indexing='ij'), -1)
    p = np.reshape(p, (-1, 2)).T
    fractal_noise_2d = fractal(p,
                              frequency=3/(nx*dx), 
                              lucanarity=2, 
                              persistence=0.7, 
                              n_octaves=5)
    fractal_noise_2d = np.reshape(fractal_noise_2d, (nx, ny))
  """
  frequencies = get_frequencies(frequency, lucanarity, n_octaves)
  amplitudes = get_amplitudes(amplitude, persistence, n_octaves)

  noise = np.zeros(p.shape[-1])

  for i_octave in range(n_octaves):
    noise += perlin(p, frequencies[i_octave], amplitudes[i_octave])
  return noise


def perlin(p, frequency=1.0, amplitude=1.0):
  """sample perlin noise in one or two dimensions

  Args:
      p (1d or 2d array): point(s) in space at which to sample fractal noise. If 
        1d, shape is (n_points). If 2d, shape is (2, n_points) where the first
        axis holds the location of each point on a 2d grid.
      frequency (float, optional): Frequency of perlin noise. Defaults to 1.0.
      amplitude (float, optional): maximum absolute amplitude of returned noise.
        Defaults to 1.0.

  Raises:
      NotImplementedError: if first axis of provided p is more than 2.

  Returns:
      1d np array: sampled noise at each point in p
  
  Example 1d usage:
    n_p, o_p, d_p = 1000, 0.0, 1
    p = np.linspace(o_p, o_p + n_p * d_p, n_p, endpoint=False)
    perlin1 = perlin(p, frequency=d_p / 50)

  Example 2d usage:
    nx = 1000
    ny = 1000
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    p = np.stack(np.meshgrid(x, y, indexing='ij'), -1)
    p = np.reshape(p, (-1, 2)).T
    perlin_noise_2d = perlin(p)
    perlin_noise_2d = np.reshape(perlin_noise_2d, (nx, ny))
  """
  dims = len(p.shape)
  if dims == 1:
    return amplitude * n(p * frequency)
  elif dims == 2:
    return amplitude * n_2d(np.array([frequency * p[0], frequency * p[1]]))
  else:
    raise NotImplementedError(
        "Dimensions of points parameter, p, is {dims} which is not supported.")


@njit(float64[:](float64[:]), parallel=True)
def fade(t):
  """get interpolation coefficient

  Samples nonlinear interpolation function that has a continuous first and 
  second derivative. It also has a zero first and second derivatives equal to
  0.0 at t=0.0 and t=1.0 which can avoid spiky artifacts in perlin noise.

  Nonlinear interp function used: 
    f(t) = 6t^5 - 15t^4 + 10t^3

  Args:
      t (1d array of float64): points which to sample f(t)

  Returns:
      1d array of float64: interplotation coeff for each point in t
  """
  return t * t * t * (t * (6 * t - 15) + 10)


@njit(float64[:](float64[:]), parallel=True)
def g(p):
  """sample the psuedo-random 1d gradients 

  Works like:
    1. cast point p to an integer
    2. use the mod operator to project the integer from 0 to HASH_TABLE_SIZE
    3. sample the HASH_TABLE to find the index within the GRADIENTS array to sample
    4. get the gradient value

  Args:
      p (1d array of float64): point(s) at which to sample the gradient function.

  Returns:
      1d array of integers: gradient values at each point in p.
  """
  return GRADIENTS[HASH_TABLE[p.astype(int64) % 256]]


@njit(float64[:](float64[:]), parallel=True)
def n(p):
  """sample perlin noise at point(s)

  Args:
      p (1d array of float64): point(s) at which to sample perlin noise

  Returns:
      1d array of float64: noise values at each point p
  """
  p0 = np.floor(p)
  p1 = p0 + 1
  t = p - p0
  f = fade(t)
  g0 = g(p0)
  g1 = g(p1)
  return (1 - f) * g0 * (p - p0) + f * g1 * (p - p1)


@njit(float64[:, :](float64[:, :]), parallel=True)
def g_2(p):
  """sample the psuedo-random 2d gradient vectors

  Works like:
    1. cast point p[0] and p[1] to integer values
    2. use the mod operator to project the integers from 0 to HASH_TABLE_SIZE and
      add them together.
    3. sample the HASH_TABLE to find the index within the GRADIENTS_2D array to sample
    4. get the gradient vector

  Args:
      p (2d array of float64): point(s) at which to sample the gradient vector
        function. shape is (2, n_points) where p[0] are the point locations on
        the first axis and p[1] are the point locations on the second axis.

  Returns:
      2d array of integers: shape of (2,n_points). Gradient vectors at each
        point in p.
  """
  return GRADIENTS_2D[HASH_TABLE[HASH_TABLE[p[0].astype(int64) % 256] +
                                 p[1].astype(int64) % 256]].T


@njit(float64[:](float64[:, :]), parallel=True)
def n_2d(p):
  """sample perlin noise at point(s) in 2 dimensions

  Args:
      p (2d array of float64): point(s) at which to sample perlin noise. Shape
        is (2, n_points) where p[0] are the point locations on the first axis
        and p[1] are the point locations on the second axis.

  Returns:
      1d array of float64: noise values at each point p
  """
  p0 = np.floor(p)
  p1 = (p0.T + np.array([1.0, 0.0])).T
  p2 = (p0.T + np.array([0.0, 1.0])).T
  p3 = (p0.T + np.array([1.0, 1.0])).T

  g0 = g_2(p0)
  g1 = g_2(p1)
  g2 = g_2(p2)
  g3 = g_2(p3)

  t0 = p[0] - p0[0]
  f0 = fade(t0)

  t1 = p[1] - p0[1]
  f1 = fade(t1)

  p0p1 = (1 - f0) * (g0 * (p - p0)).sum(0) + f0 * (g1 * (p - p1)).sum(0)
  p2p3 = (1 - f0) * (g2 * (p - p2)).sum(0) + f0 * (g3 * (p - p3)).sum(0)

  return (1 - f1) * p0p1 + f1 * p2p3


def get_frequencies(frequency, lucanarity, n_octaves):
  "create a list of frequencies for each octave"
  return _get_param_list(frequency, lucanarity, n_octaves)


def get_amplitudes(amplitude, persistence, n_octaves):
  "create a list of amplitudes for each octave"
  return _get_param_list(amplitude, persistence, n_octaves)


def _get_param_list(value, scale, n):
  """creates a list of scaled values

  If value is already a list it is returned. Otherwise a list of size n is
  created with scaled values. 

  Args:
      value (float, list of floats): value to create list from
      scale (float, list of floats): amount to scale each subsequent value in
        list
      n (int): size of returned list

  Returns:
      list of floats: list of scaled values

  Example usage one:
    value = 1
    scale = 0.5 
    n = 4
    ex = _get_param_list(value, scale, n)
    # ex returns [1, 0.5, 0.25, 0.125]
  """
  if not isinstance(value, list):
    if isinstance(scale, list):
      if n is not None:
        assert len(scale) == n
      else:
        n = len(scale)
    else:
      scale = [scale] * n
    value = [value]
    for i in range(1, n):
      value.append(value[-1] * scale[i])

  return value