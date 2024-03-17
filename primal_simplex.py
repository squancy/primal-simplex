import pprint
import numpy as np
import sympy
import scipy
import copy

def read_input():
  """
  Reads input from stdin
  
  :return: the matrix A, the vector b and the function we are optimizing for
  """

  try:
    n = int(input())
    m = int(input())
  except ValueError:
    raise ValueError("n and m must be integers")

  c = input().split()
  lines = []
  for _ in range(int(m)):
    lines.append(input())
  
  try:
    A = [] 
    for row in lines:
      a = [sympy.Rational(int(x), 1) for x in row.split()[:-1]]
      A.append(a)

    b = []
    for row in lines:
      b.append(sympy.Rational(int(row.split()[-1]), 1))

    c = [sympy.Rational(int(x), 1) for x in c]
  except ValueError:
    raise ValueError("A, b and c must be integers")

  return np.array(A), np.array(b), np.array(c)

def back_substitution(U, B):
  """
  Performs back substitution on a lower triangular matrix

  :param U: the triangular matrix
  :param B: the vector b

  :return: a vector x with the substituted values
  """

  m = U.shape[0] 
  X = np.array([sympy.Rational(0, 1) for _ in range(m)]).reshape(m, 1)

  for i in range(m - 1, -1, -1):
    X[i] = B[i]
    for j in range(i + 1, m):
      X[i] -= U[i][j] * X[j]
    if U[i][i] != 0:
      X[i] /= U[i][i]

  return X

def row_swap(A, k, l):
  """
  Performs a row swap on the matrix A

  :param A: the matrix A
  :param k: the index of a row
  :param l: the index of a row

  :return: a copy of A with the rows swapped
  """

  m = A.shape[0]
  n = A.shape[1]

  B = np.copy(A)
      
  for j in range(n):
    temp = B[k][j]
    B[k][j] = B[l][j]
    B[l][j] = temp
      
  return B

def row_scale(A, k, scale):
  """
  Scales the kth row in matrix A by scale

  :param A: the matrix A
  :param k: the kth row
  :param scale: the number by which to scale

  :return: a copy of A with the kth row scaled
  """

  m = A.shape[0]
  n = A.shape[1]

  B = np.copy(A)

  for j in range(n):
    B[k][j] *= scale
      
  return B

def row_add(A, k, l, scale):
  """
  Performs the addition of two rows in A

  :param A: the matrix A
  :param k: the index of a row
  :param l: the index of a row
  :param scale: the number by which to scale the kth row

  :return: a copy of A with the kth and lth rows added
  """

  m = A.shape[0]
  n = A.shape[1]

  B = np.copy(A)
      
  for j in range(n):
    B[l][j] += B[k][j] * scale
      
  return B

def row_reduction(A):
  """
  Performs Gaussian-elimination on an augmented matrix n x (n + 1)

  :param A: the augmented matrix A

  :return: a lower triangular matrix derived from A
  """

  m = A.shape[0]
  n = A.shape[1]

  B = np.copy(A)

  for k in range(m):
    pivot = B[k][k]
    pivot_row = k
    
    while pivot == 0 and pivot_row < m - 1:
      pivot_row += 1
      pivot = B[pivot_row][k]
        
    if pivot_row != k:
      B = row_swap(B, k, pivot_row)
        
    if pivot != 0:
      B = row_scale(B, k, sympy.Rational(1, 1) / B[k][k])
      for i in range(k + 1, m):
        B = row_add(B, k, i, -B[i][k])

  return B

def solve_lin(A, b):
  """
  Solves Ax = b
  It is assumed that a unique solution exists to the system of equations

  :param A: the matrix A
  :param b: the vector b

  :return: a vector (x_1, ..., x_n) that is the unique solution
  """

  n = A.shape[0]
  A_augmented = np.c_[A, b]
  R = row_reduction(A_augmented)
  B_reduced = R[:, n:n+1]
  A_reduced = R[:, 0:n]
  X = back_substitution(A_reduced, B_reduced)

  return X.flatten()

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
    y_prime = solve_lin(B.T, c_B)
    c_prime = y_prime@A - c
  
  while True:
    # Bijection between row indicies and x_B
    conv = {A_prime[:, ind].argmax(axis = 0): ind for ind in B_inds}
  
    print(A_prime)

    # DONE: B is optimal
    if (c_prime >= 0).all():
      return {'solvable': True, 'base': np.sort(B_inds), 'A': A_prime, 'b': b_prime, 'c': c_prime}
    
    # Select the column (p) that is appended to B
    p = select_p(c_prime, B_inds)
  
    a_p = A_prime[:, p] 
    B_inds = np.append(B_inds, p)

    # cx is not bounded above --> no solution
    if (a_p <= 0).all():
      return {'solvable': False, 'dir': a_p}

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
  res = sympy.linsolve((sympy.Matrix(A), sympy.Matrix(b)))
  if res == sympy.S.EmptySet:
    return {'solvable': False}

  ps = primal_simplex(A, b, c, B_inds, phase_one = True)
  if not ps['solvable']:
    return {'solvable': False}
  else:
    B_inds = ps['base']
    A_prime = ps['A']
    b_prime = ps['b']
    c_prime = ps['c']

    # Check if the optimum is < 0
    x = solve_lin(A[:, B_inds], b)
    print('LLL', x)
    print('binds', B_inds)
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
   
if __name__ == '__main__':
  A, b, c = read_input()
  
  # Eliminate redundant rows, if any
  _, inds = sympy.Matrix(A).T.rref()
  A = A[inds, :]
  b = b[np.array(inds)]

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
      x_B = solve_lin(B_p2, b)
      y = solve_lin(B_p2.T, c[B_inds_phase_2])

      k = 0
      x = []
      for i in range(len(A[0])):
        if i in B_inds_phase_2:
          x.append(x_B[k])
          k += 1
        else:
          x.append(0)
      
      x = np.array(x)
      print("Optimum:", y@b)
      print("Primal solution:", x)
      print("Dual solution:", y)
    else:
      print("Unbounded solution: ", res_2['dir']) 
  else:
    print("No solution")
