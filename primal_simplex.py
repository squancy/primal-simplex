import numpy as np
import sympy
import copy

def read_input():
  """
  Reads input from a file
  
  :return: the matrix A, the vector b and the function we are optimizing for
  """

  with open('inp.txt') as f:
    lines = [line.rstrip() for line in f]
    m = int(lines[1])
    c = np.array([sympy.Rational(x, 1) for x in lines[2].split(' ')])
    A = np.array([[sympy.Rational(x, 1) for x in lines[3 + i].split(' ')] for i in range(m)])
    b = A[:, -1]
    A = A[:, :-1]
  print(A)
  return A, b, c

def pivot(A_prime, b_prime, c_prime, r, p):
  """
  Performs a pivot operation on a simplex table

  :param A_prime: the matrix A
  :param b_prime: the vector b
  :param c_prime: the vector c
  :param r: the row used for pivoting
  :param p: the column used for pivoting
  """

  A_prime_v = copy.deepcopy(A_prime)
  b_prime_v = copy.deepcopy(b_prime)
  c_prime_v = copy.deepcopy(c_prime)

  a_rp = A_prime[r][p]
  a_rj_prime = A_prime[r] / a_rp
  A_prime_v[r] = a_rj_prime
  
  for i in range(len(b_prime)):
    if i == r: continue
    b_prime_v[i] -= b_prime[r] * (A_prime[i][p] / a_rp) 
  b_prime_v[r] /= a_rp

  for i in range(len(A_prime)):
    arr = []
    for j in range(len(A_prime[i])):
      if i == r: continue
      arr.append(A_prime[i][j] - A_prime[r][j] * (A_prime[i][p] / a_rp))
    if i != r:
      A_prime_v[i, :] = np.array(arr)
  
  c_prime_v = np.array([c_prime[j] - A_prime[r][j] * (c_prime[p] / a_rp) for j in range(len(c_prime))])
  return A_prime_v, b_prime_v, c_prime_v

def select_p_art(B_inds, A_max_ind):
  """
  Select a column p from the base with an artificial variable 

  :param B_inds: the column indicies representing B
  :param A_max_ind: starting column index for artificial variables

  :return: a column index p from B with an artificial variable
  """

  return B_inds[np.where(B_inds >= A_max_ind)][0] 

def select_p(c_prime, B_inds):
  """
  Selects p: the first non-base index for which c_p < 0 is satisfied

  :param c_prime: the vector c in the simplex table
  :param B_inds: the column indicies representing B
  
  :return: the first non-base index for which c_p < 0 is satisfied
  """

  for i in range(len(c_prime)):
    if c_prime[i] < 0 and not i in B_inds:
      return i

def select_r_art(A_prime, A_max_ind, conv2, p):
  """
  Selects the first index in the row of p where the value is not zero

  :param A_prime: the matrix A in the simplex table
  :param A_max_ind: starting column index for artificial variables
  :param conv2: dictionary that converts column index p to row index r
  :param p: column index p
  
  :return: first index in the row of p where the value is not zero
  """

  a_r = A_prime[conv2[p]][:A_max_ind]
  return np.where(a_r != 0)[0][0]

def select_r(b_prime, a_p):
  """
  Selects the row r that is taken out of B

  :param b_prime: the vector b in the simplex table
  :param a_p: pth column of the matrix A in the simplex table

  :return: r such that b_r = a_ir = min(b_i / a_ip), a_ip > 0
  """

  tmp = np.array([float('inf') if a_p[i] <= 0 else b_prime[i] / a_p[i] for i in range(len(a_p))])
  return tmp.argmin(axis = 0)

def I_solve(I, b):
  """
  Solves an equation Ax = b where the columns of A are the permuted columns of the identity matrix

  :param I: the matrix A
  :param b: the vector b

  :return: a vector x which is the solution
  """

  x = [0] * I.shape[0]
  for i in range(I.shape[1]):
    pos = np.where(I[:, i] == 1)[0][0]
    x[i] = b[pos]
  return np.array(x)

def primal_simplex(A, b, c, B_inds, phase_one = False):
  """
  Performs the primal simplex algorithm 

  :param A: the matrix A
  :param b: the vector b
  :param c: the vector that is used for finding max cx
  :param B_inds: the column indicies of A that form the base B

  :return: a both primal and dual feasible base if max cx is bounded or a direction
  """

  B = A[:, B_inds]
  if phase_one:
    A_prime = copy.deepcopy(A)
    b_prime = copy.deepcopy(b)
    c_prime = copy.deepcopy(c)
  else:
    c_B = c[B_inds]
    B_inv = np.array(sympy.Matrix(B).inv())
    A_prime = B_inv@A
    b_prime = B_inv@b
    c_prime = c_B@A_prime - c
  
  while True:
    # Bijection between row indicies and x_B
    conv = {A_prime[:, ind].argmax(axis = 0): ind for ind in B_inds}
  
    print('A', A_prime)
    print('b', b_prime)
    print('c', c_prime)	
    print()

    # DONE: B is optimal
    if (c_prime >= 0).all():
      return {'solvable': True, 'base': np.sort(B_inds), 'A': A_prime, 'b': b_prime, 'c': c_prime}
    
    # Select the column (p) that is appended to B
    p = select_p(c_prime, B_inds)
  
    a_p = A_prime[:, p] 

    # cx is not bounded above --> no solution
    if (a_p <= 0).all():
      return {'solvable': False, 'dir': extend_x(B_inds, A, I_solve(A_prime[:, B_inds], b_prime))}

    B_inds = np.append(B_inds, p)

    # Select the row (r) that is taken out of B
    r = select_r(b_prime, a_p)
    B_inds = np.delete(B_inds, np.argwhere(B_inds == conv[r]))
    
    # Calculate the new simplex table
    A_prime, b_prime, c_prime = pivot(A_prime, b_prime, c_prime, r, p)
    
def phase_one(A, b, c, A_max_ind, B_inds):
  """
  Performs the first phase of the algorithm
  
  :param A: the matrix A
  :param b: the vector b
  :return: either an error if the polyhedron is empty or a starting base B
  """

  # Check if Ax = b is solvable at all
  """
  res = sympy.linsolve((sympy.Matrix(A), sympy.Matrix(b)))
  if res == sympy.S.EmptySet:
    return {'solvable': False}
  """

  ps = primal_simplex(A, b, c, B_inds, phase_one = True)
  if not ps['solvable']:
    return {'solvable': False}
  else:
    B_inds = ps['base']
    A_prime = ps['A']
    b_prime = ps['b']
    c_prime = ps['c']

    # Check if the optimum is < 0
    x = I_solve(A_prime[:, B_inds], b_prime)
    if any([x[i] != 0 and B_inds[i] >= A_max_ind for i in range(len(x))]):
      return {'solvable': False}

    # Check for artificial variables in the base
    while len(set(B_inds) & set(range(A_max_ind, len(A[0])))) > 0:
      conv2 = {ind: A_prime[:, ind].argmax(axis = 0) for ind in B_inds}

      # Select the column (p) that is appended to B
      p = select_p_art(B_inds, A_max_ind)
    
      a_p = A_prime[:, p] 
      B_inds = np.delete(B_inds, np.argwhere(B_inds == p))

      # Select the row (r) that is taken out of B
      r = select_r_art(A_prime, A_max_ind, conv2, p)
      B_inds = np.append(B_inds, r)
      
      # Calculate the new simplex table
      A_prime, b_prime, c_prime = pivot(A_prime, b_prime, c_prime, r, p)

    return {'solvable': True, 'base': sorted(B_inds)}
  
def identity(n):
  """
  Creates an n x n identity matrix

  :param n: size of the matrix
  :return: n x n identity matrix
  """

  res = []
  for i in range(n):
    res.append([sympy.Rational(1, 1) if i == j else sympy.Rational(0, 1) for j in range(n)])
  return np.array(res)

def extend_x(B_inds, A, x_B):
  k = 0
  x = []
  for i in range(len(A[0])):
    if i in B_inds:
      x.append(x_B[k])
      k += 1
    else:
      x.append(0)
  return x

def solve():
  A, b, c = read_input()

  B_inds = np.array([i for i in range(len(A[0]), len(A[0]) + len(A))])
  I = identity(len(A))
  A_extended = np.hstack((A, I))
  c_extended = np.hstack((-A.sum(axis = 0), np.array([0] * len(A))))
  res = phase_one(A_extended, b, c_extended, len(A[0]), B_inds)
  if res['solvable']:
    B_inds = res['base']
    res_2 = primal_simplex(A, b, c, B_inds)
    if res_2['solvable']:
      B_inds_phase_2 = res_2['base']
      B_p2 = A[:, B_inds_phase_2]
      x_B = I_solve(res_2['A'][:, B_inds_phase_2], res_2['b'])
      y = c[B_inds_phase_2]@np.array(sympy.Matrix(A[:, B_inds_phase_2]).inv())
      x = np.array(extend_x(B_inds_phase_2, A, x_B))
      
      print("Optimum:", y@b)
      print("Primal solution:", x)
      print("Dual solution:", y)
    else:
      print("Unbounded solution: ", res_2['dir']) 
  else:
    print("No solution")


if __name__ == '__main__':
  solve()
