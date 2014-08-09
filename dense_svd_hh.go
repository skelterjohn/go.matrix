package matrix

import (
	"errors"
	"fmt"
	"math"
)

func (A *DenseMatrix) SVDHH() (U, S, V *DenseMatrix, err error) {
	maxiter := 150
	epsilon := 1e-300

	// Get the number of rows/columns of the matrix
	rows := A.Rows()
	cols := A.Cols()

	U = A.Copy()
	S = Zeros(cols, cols)
	V = Zeros(cols, cols)
	tmp := make([]float64, cols)

	i, its, j, jj, k := -1, -1, -1, -1, -1
	nm, ppi := 0, 0
	var flag bool
	var anorm, c, f, h, s, scale, x, y, z, g float64
	anorm, c, f, h, s, scale, x, y, z, g = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

	anorm_nonzero := false

	// Householder reduction to bidiagonal form.
	for i = 0; i < cols; i++ {
		ppi = i + 1
		tmp[i] = scale * g
		g, s, scale = 0, 0, 0
		if i < rows {
			// compute the sum of the i-th column, starting from the i-th row
			for k = i; k < rows; k++ {
				scale += math.Abs(U.Get(k, i))
			}
			if math.Abs(scale) > epsilon {
				// multiply the i-th column by 1.0/scale, start from the i-th element
				// sum of squares of column i, start from the i-th element
				for k = i; k < rows; k++ {
					val := U.Get(k, i) / scale
					U.Set(k, i, val)
					s += val * val
				}
				f = U.Get(i, i) // f is the diag elem
				if !(s >= 0) {
					return nil, nil, nil, errors.New("SVDHH: Sum of squares is negative [1].")
				}
				g = -copySign(math.Sqrt(s), f)
				h = f*g - s
				U.Set(i, i, f-g)
				for j = ppi; j < cols; j++ {
					// dot product of columns i and j, starting from the i-th row
					for s, k = 0, i; k < rows; k++ {
						s += U.Get(k, i) * U.Get(k, j)
					}
					if !(h != 0) {
						return nil, nil, nil, errors.New("SVDHH: Zero denominator [1].")
					}
					f = s / h
					// copy the scaled i-th column into the j-th column
					for k = i; k < rows; k++ {
						val := U.Get(k, j) + f*U.Get(k, i)
						U.Set(k, j, val)
					}
				}
				for k = i; k < rows; k++ {
					U.Set(k, i, U.Get(k, i)*scale)
				}
			}
		}
		// save singular value
		S.Set(i, i, scale*g)
		g, s, scale = 0, 0, 0
		if (i < rows) && (i+1 != cols) {
			// sum of row i, start from columns i+1
			for k = ppi; k < cols; k++ {
				scale += math.Abs(U.Get(i, k))
			}
			if math.Abs(scale) > epsilon {
				for k = ppi; k < cols; k++ {
					val := U.Get(i, k) / scale
					U.Set(i, k, val)
					s += val * val
				}
				f = U.Get(i, ppi)
				if !(s >= 0) {
					return nil, nil, nil, errors.New("SVDHH: Sum of squares is negative [2].")
				}
				g = -copySign(math.Sqrt(s), f)
				h = f*g - s
				U.Set(i, ppi, f-g)
				if !(h != 0) {
					return nil, nil, nil, errors.New("SVDHH: Zero denominator [2].")
				}
				for k = ppi; k < cols; k++ {
					tmp[k] = U.Get(i, k) / h
				}
				for j = ppi; j < rows; j++ {
					for s, k = 0, ppi; k < cols; k++ {
						s += U.Get(j, k) * U.Get(i, k)
					}
					for k = ppi; k < cols; k++ {
						U.Set(j, k, U.Get(j, k)+s*tmp[k])
					}
				}
				for k = ppi; k < cols; k++ {
					U.Set(i, k, U.Get(i, k)*scale)
				}
			}
		}
		anorm_nonzero = anorm_nonzero || (math.Abs(S.Get(i, i))+math.Abs(tmp[i]) != 0)
	}
	if anorm_nonzero {
		anorm = 1
	}
	// Accumulation of right-hand transformations.
	for i = cols - 1; i >= 0; i-- {
		if i < cols-1 {
			if math.Abs(g) > epsilon {
				if !(U.Get(i, ppi) != 0) {
					return nil, nil, nil, errors.New("SVDHH: Unfortunate zero in U.")
				}
				for j = ppi; j < cols; j++ {
					V.Set(j, i, (U.Get(i, j)/U.Get(i, ppi))/g)
				}
				for j = ppi; j < cols; j++ {
					for s, k = 0, ppi; k < cols; k++ {
						s += U.Get(i, k) * V.Get(k, j)
					}
					for k = ppi; k < cols; k++ {
						V.Set(k, j, V.Get(k, j)+s*V.Get(k, i))
					}
				}
			}
			for j = ppi; j < cols; j++ {
				V.Set(i, j, 0)
				V.Set(j, i, 0)
			}
		}
		V.Set(i, i, 1)
		g = tmp[i]
		ppi = i
	}
	// Accumulation of left-hand transformations.
	if cols < rows {
		i = cols - 1
	} else {
		i = rows - 1
	}
	for ; i >= 0; i-- {
		ppi = i + 1
		g = S.Get(i, i)
		for j = ppi; j < cols; j++ {
			U.Set(i, j, 0)
		}
		if math.Abs(g) > epsilon {
			g = 1 / g
			for j = ppi; j < cols; j++ {
				for s, k = 0, ppi; k < rows; k++ {
					s += U.Get(k, i) * U.Get(k, j)
				}
				if !(U.Get(i, i) != 0) {
					return nil, nil, nil, errors.New("SVDHH: Zero in diagonal of U.")
				}
				f = (s / U.Get(i, i)) * g
				for k = i; k < rows; k++ {
					U.Set(k, j, U.Get(k, j)+f*U.Get(k, i))
				}
			}
			for j = i; j < rows; j++ {
				U.Set(j, i, U.Get(j, i)*g)
			}
		} else {
			for j = i; j < rows; j++ {
				U.Set(j, i, 0)
			}
		}
		U.Set(i, i, U.Get(i, i)+1)
	}

	// Diagonalization of the bidiagonal form.
	for k = cols - 1; k >= 0; k-- { // Loop over singular values.
		for its = 1; its <= maxiter; its++ { // Loop over allowed iterations.
			flag = true
			for ppi = k; ppi >= 0; ppi-- { // Test for splitting.
				nm = ppi - 1 // Note that tmp[1] is always zero.
				if (math.Abs(tmp[ppi]) + anorm) == anorm {
					flag = false
					break
				}
				if math.Abs(S.Get(nm, nm)+anorm) == anorm {
					break
				}
			}
			if flag {
				c = 0.0 // Cancellation of tmp[l], if l>1:
				s = 1.0
				for i = ppi; i <= k; i++ {
					f = s * tmp[i]
					tmp[i] = c * tmp[i]
					if (math.Abs(f) + anorm) == anorm {
						break
					}
					g = S.Get(i, i)
					h = pythag(f, g)
					S.Set(i, i, h)
					if !(h != 0) {
						return nil, nil, nil, errors.New("SVDHH: Zero denominator. [3]")
					}
					h = 1.0 / h
					c = g * h
					s = (-f * h)
					for j = 0; j < rows; j++ {
						y = U.Get(j, nm)
						z = U.Get(j, i)
						U.Set(j, nm, y*c+z*s)
						U.Set(j, i, z*c-y*s)
					}
				}
			}
			z = S.Get(k, k)

			if ppi == k { // Convergence.
				if z < 0 { // Singular value is made nonnegative.
					S.Set(k, k, -z)
					for j = 0; j < cols; j++ {
						V.Set(j, k, -V.Get(j, k))
					}
				}
				break
			}

			x = S.Get(ppi, ppi) // Shift from bottom 2-by-2 minor:
			nm = k - 1
			y = S.Get(nm, nm)
			g = tmp[nm]
			h = tmp[k]
			if !(h != 0 && y != 0) {
				return nil, nil, nil, errors.New("SVDHH: Zero denominator [4].")
			}
			f = ((y-z)*(y+z) + (g-h)*(g+h)) / (2.0 * h * y)

			g = pythag(f, 1.0)
			if !(x != 0) {
				return nil, nil, nil, errors.New("SVDHH: Zero denominator [5].")
			}
			if !((f + copySign(g, f)) != 0) {
				return nil, nil, nil, errors.New("SVDHH: Zero denominator [6].")
			}
			f = ((x-z)*(x+z) + h*((y/(f+copySign(g, f)))-h)) / x

			// Next QR transformation:
			c, s = 1, 1
			for j = ppi; j <= nm; j++ {
				i = j + 1
				g = tmp[i]
				y = S.Get(i, i)
				h = s * g
				g = c * g
				z = pythag(f, h)
				tmp[j] = z
				if !(z != 0) {
					return nil, nil, nil, errors.New("Zero denominator [7].")
				}
				c = f / z
				s = h / z
				f = x*c + g*s
				g = g*c - x*s
				h = y * s
				y = y * c
				for jj = 0; jj < cols; jj++ {
					x = V.Get(jj, j)
					z = V.Get(jj, i)
					V.Set(jj, j, x*c+z*s)
					V.Set(jj, i, z*c-x*s)
				}
				z = pythag(f, h)
				S.Set(j, j, z)
				if math.Abs(z) > epsilon {
					z = 1.0 / z
					c = f * z
					s = h * z
				}
				f = (c * g) + (s * y)
				x = (c * y) - (s * g)
				for jj = 0; jj < rows; jj++ {
					y = U.Get(jj, j)
					z = U.Get(jj, i)
					U.Set(jj, j, y*c+z*s)
					U.Set(jj, i, z*c-y*s)
				}
			}
			tmp[ppi] = 0.0
			tmp[k] = f
			S.Set(k, k, x)
		}
	}

	//Sort eigen values:
	for i = 0; i < cols; i++ {

		S_max := S.Get(i, i)
		i_max := i
		for j = i + 1; j < cols; j++ {
			Sj := S.Get(j, j)
			if Sj > S_max {
				S_max = Sj
				i_max = j
			}
		}
		if i_max != i {
			// swap eigenvalues
			tmpi := S.Get(i, i)
			S.Set(i, i, S.Get(i_max, i_max))
			S.Set(i_max, i_max, tmpi)

			// swap eigenvectors
			swapCols(U, i, i_max)
			swapCols(V, i, i_max)
		}
	}

	if its == maxiter {
		return nil, nil, nil, fmt.Errorf("SVDHH: did not converge after %d iterations.", maxiter)
	} else {
		return U, S, V, nil
	}
}

func swapCols(m *DenseMatrix, ci, cj int) {
	rows := m.Rows()
	for r := 0; r < rows; r++ {
		tmp := m.Get(r, cj)
		m.Set(r, cj, m.Get(r, ci))
		m.Set(r, ci, tmp)
	}
}

func pythag(a, b float64) float64 {
	var ct float64
	at := math.Abs(a)
	bt := math.Abs(b)
	if at > bt {
		ct = bt / at
		return at * math.Sqrt(1.0+ct*ct)
	} else {
		if bt == 0 {
			return 0.0
		} else {
			ct = at / bt
			return bt * math.Sqrt(1.0+ct*ct)
		}
	}
}

func copySign(a, b float64) float64 {
	if b >= 0 {
		return math.Abs(a)
	} else {
		return -math.Abs(a)
	}
}
