{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "1505ad73-a15f-42fe-a08f-d5129f709995",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Minimum value found at x = -0.33333152943479877, y = 0.33333152943479877\n",
      "Minimum value of the function is 0.6666666666764288\n"
     ]
    }
   ],
   "source": [
    "#k\n",
    "import numpy as np\n",
    "\n",
    "# Define the function f(x, y)\n",
    "def f(x, y):\n",
    "    return x**2 + y**2 - x*y + x - y + 1\n",
    "\n",
    "# Define the gradients of f(x, y)\n",
    "def grad_f(x, y):\n",
    "    df_dx = 2*x - y + 1\n",
    "    df_dy = 2*y - x - 1\n",
    "    return np.array([df_dx, df_dy])\n",
    "\n",
    "# Gradient Descent function\n",
    "def gradient_descent(learning_rate, initial_guess, max_iters, tolerance):\n",
    "    x, y = initial_guess\n",
    "    for _ in range(max_iters):\n",
    "        grad = grad_f(x, y)\n",
    "        new_x = x - learning_rate * grad[0]\n",
    "        new_y = y - learning_rate * grad[1]\n",
    "        \n",
    "        if np.sqrt((new_x - x)**2 + (new_y - y)**2) < tolerance:\n",
    "            break\n",
    "        \n",
    "        x, y = new_x, new_y\n",
    "    \n",
    "    return x, y\n",
    "\n",
    "# Parameters\n",
    "learning_rate = 0.1\n",
    "initial_guess = (0, 0)\n",
    "max_iters = 1000\n",
    "tolerance = 1e-6\n",
    "\n",
    "# Run gradient descent\n",
    "min_x, min_y = gradient_descent(learning_rate, initial_guess, max_iters, tolerance)\n",
    "\n",
    "print(f\"Minimum value found at x = {min_x}, y = {min_y}\")\n",
    "print(f\"Minimum value of the function is {f(min_x, min_y)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "0d907597-7cae-4d9b-8a2e-2b8b645c9808",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Function: 2*x + 1\n",
      "Derivative: 2\n",
      "Integral value: 66.66666666666669\n",
      "Absolute error estimate: 7.401486830834379e-13\n",
      "Fitted parameters: a = 1.9730383453307772 , b = 3.729365701834092\n"
     ]
    },
    {
     "data":
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Coefficients: [2.         2.16666667]\n",
      "Intercept: -3.166666666666667\n"
     ]
    },
    {
     "data":
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data":
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#C\n",
    "# Differntition\n",
    "\n",
    "from scipy.interpolate import interp1d\n",
    "from sklearn.linear_model import LinearRegression\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.optimize import curve_fit\n",
    "from scipy.integrate import quad\n",
    "import numpy as np\n",
    "import sympy as sp\n",
    "\n",
    "# Define the function\n",
    "x = sp.Symbol('x')\n",
    "f = x*3 + 2*x*2 - 5*x + 1\n",
    "\n",
    "# Differentiate the function\n",
    "derivative = sp.diff(f, x)\n",
    "\n",
    "print(\"Function:\", f)\n",
    "print(\"Derivative:\", derivative)\n",
    "\n",
    "# Numerical intergration\n",
    "\n",
    "# Define the function\n",
    "\n",
    "\n",
    "def f(x):\n",
    "    return x**2 + 2*x\n",
    "\n",
    "\n",
    "# Integrate the function from 0 to 5\n",
    "result, error = quad(f, 0, 5)\n",
    "\n",
    "print(\"Integral value:\", result)\n",
    "print(\"Absolute error estimate:\", error)\n",
    "\n",
    "# Curve Fitting\n",
    "\n",
    "# Generate sample data\n",
    "x = np.linspace(0, 10, 50)\n",
    "y = 2*x + 3 + np.random.normal(0, 2, 50)\n",
    "\n",
    "# Define the curve fitting function\n",
    "\n",
    "\n",
    "def linear_func(x, a, b):\n",
    "    return a*x + b\n",
    "\n",
    "\n",
    "# Perform curve fitting\n",
    "popt, pcov = curve_fit(linear_func, x, y)\n",
    "\n",
    "# Print the fitted parameters\n",
    "print(\"Fitted parameters: a =\", popt[0], \", b =\", popt[1])\n",
    "\n",
    "# Plot the data and fitted curve\n",
    "plt.scatter(x, y, label='Data')\n",
    "plt.plot(x, linear_func(x, *popt), color='r', label='Fitted Curve')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "\n",
    "# Linear Regression\n",
    "\n",
    "# Generate sample data\n",
    "x = np.array([[1, 2], [1, 4], [2, 2], [2, 4], [3, 5]])\n",
    "y = np.array([3, 7, 5, 11, 13])\n",
    "\n",
    "# Create a linear regression model\n",
    "model = LinearRegression()\n",
    "\n",
    "# Fit the model to the data\n",
    "model.fit(x, y)\n",
    "\n",
    "# Print the coefficients and intercept\n",
    "print(\"Coefficients:\", model.coef_)\n",
    "print(\"Intercept:\", model.intercept_)\n",
    "\n",
    "# Make predictions\n",
    "predictions = model.predict(x)\n",
    "\n",
    "# Plot the data and regression line\n",
    "plt.scatter(x[:, 0], y, label='Data')\n",
    "plt.plot(x[:, 0], predictions, color='r', label='Regression Line')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "\n",
    "# Spine interpolation\n",
    "\n",
    "\n",
    "# Generate sample data\n",
    "x = np.array([1, 2, 3, 4, 5])\n",
    "y = np.array([2, 5, 8, 5, 3])\n",
    "\n",
    "# Create a spline interpolation object\n",
    "spline = interp1d(x, y, kind='cubic')\n",
    "\n",
    "# Generate new x values for interpolation\n",
    "x_new = np.linspace(1, 5, 100)\n",
    "\n",
    "# Perform spline interpolation\n",
    "y_new = spline(x_new)\n",
    "\n",
    "# Plot the original data and interpolated curve\n",
    "plt.scatter(x, y, label='Data')\n",
    "plt.plot(x_new, y_new, color='r', label='Spline Interpolation')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "cb446430-8a2e-4d6b-a9ca-962027385322",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Approximate integral of sin(x) from 0 to 3.141592653589793 using 10 trapezoids is 1.9835235375094546\n"
     ]
    },
    {
     "data":
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#G\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def f(x):\n",
    "    return np.sin(x)  # Example function\n",
    "\n",
    "def trapezoidal_rule(a, b, n):\n",
    "    h = (b - a) / n\n",
    "    x = np.linspace(a, b, n+1)\n",
    "    y = f(x)\n",
    "    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:n]) + y[n])\n",
    "    return integral\n",
    "\n",
    "# Parameters\n",
    "a = 0  # Start of the interval\n",
    "b = np.pi  # End of the interval\n",
    "n = 10  # Number of trapezoids\n",
    "\n",
    "# Calculate the integral\n",
    "integral = trapezoidal_rule(a, b, n)\n",
    "\n",
    "# Print the result\n",
    "print(f\"Approximate integral of sin(x) from {a} to {b} using {n} trapezoids is {integral}\")\n",
    "\n",
    "# Plotting\n",
    "x = np.linspace(a, b, 1000)\n",
    "y = f(x)\n",
    "\n",
    "plt.plot(x, y, 'b', label='sin(x)')\n",
    "plt.fill_between(x, 0, y, color='skyblue', alpha=0.4)\n",
    "\n",
    "# Plot trapezoids\n",
    "x_trap = np.linspace(a, b, n+1)\n",
    "y_trap = f(x_trap)\n",
    "\n",
    "for i in range(n):\n",
    "    plt.plot([x_trap[i], x_trap[i], x_trap[i+1], x_trap[i+1]], [0, y_trap[i], y_trap[i+1], 0], 'r')\n",
    "\n",
    "plt.title('Trapezoidal Rule Approximation')\n",
    "plt.xlabel('x')\n",
    "plt.ylabel('f(x)')\n",
    "plt.legend()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "3a2481ad-3c97-4568-ae6c-5e935561aee6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Lagrange polynomial coefficients: [ 3.42633673e-14 -2.68014957e-14  1.00000000e+00 -5.97595060e-16]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "def lagrange_interpolation(x, y):\n",
    "    def L(i, x_val):\n",
    "        n = len(x)\n",
    "        result = 1.0\n",
    "        for j in range(n):\n",
    "            if j != i:\n",
    "                result *= (x_val - x[j]) / (x[i] - x[j])\n",
    "        return result\n",
    "    \n",
    "    def P(x_val):\n",
    "        n = len(x)\n",
    "        return sum(y[i] * L(i, x_val) for i in range(n))\n",
    "    \n",
    "    # Compute coefficients\n",
    "    coeffs = np.polyfit(x, [P(xi) for xi in x], len(x)-1)\n",
    "    return coeffs[::-1]  # Reverse to get ascending order of powers\n",
    "\n",
    "# Example usage\n",
    "x = [1, 2, 3, 4]\n",
    "y = [1, 4, 9, 16]\n",
    "coefficients = lagrange_interpolation(x, y)\n",
    "print(\"Lagrange polynomial coefficients:\", coefficients)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "b8ed3581-b800-4efb-ba34-adffad2393d0",
   "metadata": {},
   "outputs":
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "def newton_divided_difference(x, y):\n",
    "    n = len(x)\n",
    "    coef = np.zeros([n, n])\n",
    "    coef[:,0] = y\n",
    "    \n",
    "    for j in range(1,n):\n",
    "        for i in range(n-j):\n",
    "            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])\n",
    "\n",
    "    return coef[0]\n",
    "\n",
    "def newton_interpolation(x, y):\n",
    "    coef = newton_divided_difference(x, y)\n",
    "    \n",
    "    def P(x_val):\n",
    "        n = len(x) - 1\n",
    "        p = coef[n]\n",
    "        for k in range(1, n+1):\n",
    "            p = coef[n-k] + (x_val - x[n-k])*p\n",
    "        return p\n",
    "    \n",
    "    return P\n",
    "\n",
    "# Given data points\n",
    "x = [1, 2, 3, 4]\n",
    "y = [1, 4, 9, 16]\n",
    "\n",
    "# Create the interpolating polynomial\n",
    "P = newton_interpolation(x, y)\n",
    "\n",
    "# Test the polynomial\n",
    "x_test = np.linspace(1, 4, 100)\n",
    "y_test = [P(xi) for xi in x_test]\n",
    "\n",
    "# Plot the results\n",
    "import matplotlib.pyplot as plt\n",
    "plt.plot(x, y, 'ro', label='Data points')\n",
    "plt.plot(x_test, y_test, label=\"Newton's polynomial\")\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "01ddf8ee-2cc6-44ab-bad6-a30ac9ba2ea1",
   "metadata": {},
   "outputs":
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Polynomial coefficients (highest degree first):\n",
      "[   4.105625    -47.96069444  222.25979167 -362.74531746  191.125     ]\n"
     ]
    }
   ],
   "source": [
    "#The output would be a graph showing the original data points and the fitted polynomial curve.\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Define the data points\n",
    "x = np.array([1, 2, 3, 4, 5, 6])\n",
    "y = np.array([5.5, 43.1, 128, 290.7, 498.4, 978.67])\n",
    "\n",
    "# Fit a 4th degree polynomial\n",
    "p = np.polyfit(x, y, 4)\n",
    "\n",
    "# Create a smoother set of x-values for the fitted curve\n",
    "x2 = np.linspace(1, 6, 51)  # 51 points from 1 to 6\n",
    "\n",
    "# Evaluate the fitted polynomial at the new x-values\n",
    "y2 = np.polyval(p, x2)\n",
    "\n",
    "# Create the plot\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(x, y, 'o', label='Original data')  # Original points\n",
    "plt.plot(x2, y2, '-', label='Polynomial fit')  # Fitted curve\n",
    "plt.grid(True)\n",
    "plt.xlabel('x')\n",
    "plt.ylabel('y')\n",
    "plt.title('4th Degree Polynomial Fit')\n",
    "plt.legend()\n",
    "\n",
    "# Show the plot\n",
    "plt.show()\n",
    "\n",
    "# Print the polynomial coefficients\n",
    "print(\"Polynomial coefficients (highest degree first):\")\n",
    "print(p)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "af919769-2616-4828-a87c-3e0f2dd5d184",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The approximate integral of x^2 from 0 to 1 is: 0.333333\n"
     ]
    }
   ],
   "source": [
    "def trapezoidal_rule(f, a, b, n):\n",
    "    h = (b - a) / n\n",
    "    sum = 0.5 * (f(a) + f(b))\n",
    "    for i in range(1, n):\n",
    "        x = a + i * h\n",
    "        sum += f(x)\n",
    "    return h * sum\n",
    "\n",
    "# Example usage\n",
    "def f(x):\n",
    "    return x**2  # Example function to integrate\n",
    "\n",
    "a, b = 0, 1  # Integration limits\n",
    "n = 1000  # Number of subintervals\n",
    "\n",
    "result = trapezoidal_rule(f, a, b, n)\n",
    "print(f\"The approximate integral of x^2 from {a} to {b} is: {result:.6f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "9f0f3869-8841-4c6d-bb75-6fdcce5be7f2",
   "metadata": {},
   "outputs": 
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def analyze_signal(f1, f2, sample_rate, duration):\n",
    "    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)\n",
    "    signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)\n",
    "    \n",
    "    fft_result = np.fft.fft(signal)\n",
    "    frequencies = np.fft.fftfreq(len(t), 1/sample_rate)\n",
    "    \n",
    "    plt.figure(figsize=(12, 6))\n",
    "    plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result)[:len(frequencies)//2])\n",
    "    plt.title('Frequency Spectrum')\n",
    "    plt.xlabel('Frequency (Hz)')\n",
    "    plt.ylabel('Magnitude')\n",
    "    plt.grid(True)\n",
    "    plt.show()\n",
    "\n",
    "# Parameters\n",
    "f1, f2 = 50, 120  # Hz\n",
    "sample_rate = 1000  # Hz\n",
    "duration = 1  # second\n",
    "\n",
    "analyze_signal(f1, f2, sample_rate, duration)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "cb0c4e15-4c93-41be-938f-572dad2d2f51",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 1: x = 0.016433, f(x) = 0.000359, error = 5.085193\n",
      "Iteration 2: x = 0.094298, f(x) = -0.000229, error = 0.825731\n",
      "Iteration 3: x = 0.042656, f(x) = 0.000177, error = 1.210688\n",
      "Final approximation: x = 0.042656\n"
     ]
    }
   ],
   "source": [
    "def f(x):\n",
    "    return x**3 - 0.165*x**2 + 3.993e-4\n",
    "\n",
    "def f_prime(x):\n",
    "    return 3*x**2 - 0.33*x\n",
    "\n",
    "def newton_method(x0, iterations):\n",
    "    x = x0\n",
    "    for i in range(iterations):\n",
    "        fx = f(x)\n",
    "        fpx = f_prime(x)\n",
    "        x_new = x - fx / fpx\n",
    "        error = abs((x_new - x) / x_new)\n",
    "        \n",
    "        print(f\"Iteration {i+1}: x = {x_new:.6f}, f(x) = {f(x_new):.6f}, error = {error:.6f}\")\n",
    "        \n",
    "        x = x_new\n",
    "    \n",
    "    return x\n",
    "\n",
    "# Initial guess\n",
    "x0 = 0.1\n",
    "\n",
    "# Perform 3 iterations\n",
    "result = newton_method(x0, 3)\n",
    "print(f\"Final approximation: x = {result:.6f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "4cce7fc4-4c5c-4af9-8852-5fdaff117949",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The interpolated y value at x = 4.0 is: 7.1111\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "def linear_interpolation(x, x0, x1, y0, y1):\n",
    "    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)\n",
    "\n",
    "# Given data\n",
    "x = np.array([2.00, 4.25, 5.25, 7.81, 9.20, 10.60])\n",
    "y = np.array([7.2, 7.1, 6.0, 5.0, 3.5, 5.0])\n",
    "\n",
    "# Find the appropriate interval for x = 4.0\n",
    "for i in range(len(x) - 1):\n",
    "    if x[i] <= 4.0 <= x[i+1]:\n",
    "        x0, x1 = x[i], x[i+1]\n",
    "        y0, y1 = y[i], y[i+1]\n",
    "        break\n",
    "\n",
    "# Calculate y at x = 4.0\n",
    "y_interpolated = linear_interpolation(4.0, x0, x1, y0, y1)\n",
    "\n",
    "print(f\"The interpolated y value at x = 4.0 is: {y_interpolated:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "475abdd1-4adb-4296-a91d-1133099c9a66",
   "metadata": {},
   "outputs": 
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Value at x = 2.5: 1.575\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from scipy import interpolate\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Generate some data\n",
    "x = np.array([0, 1, 2, 3, 4, 5])\n",
    "y = np.array([0, 2, 1, 3, 7, 10])\n",
    "\n",
    "# Create cubic spline\n",
    "cs = interpolate.CubicSpline(x, y)\n",
    "\n",
    "# Generate points for smooth curve\n",
    "xs = np.linspace(0, 5, 100)\n",
    "ys = cs(xs)\n",
    "\n",
    "# Plot\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(x, y, 'o', label='Data points')\n",
    "plt.plot(xs, ys, label='Cubic Spline')\n",
    "plt.legend()\n",
    "plt.title('Cubic Spline Interpolation')\n",
    "plt.xlabel('x')\n",
    "plt.ylabel('y')\n",
    "plt.grid(True)\n",
    "plt.show()\n",
    "\n",
    "# Evaluate spline at a specific point\n",
    "x_eval = 2.5\n",
    "y_eval = cs(x_eval)\n",
    "print(f\"Value at x = {x_eval}: {y_eval}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "f0e2f97c-b1a1-4b42-826b-5dd489990945",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Trapezoidal Rule: 1.9998355038874436\n",
      "Simpson's Rule: 2.0000000108245035\n",
      "Actual value: 2.0\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "def trapezoidal_rule(f, a, b, n):\n",
    "    x = np.linspace(a, b, n+1)\n",
    "    y = f(x)\n",
    "    return (b - a) / (2 * n) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])\n",
    "\n",
    "def simpson_rule(f, a, b, n):\n",
    "    if n % 2 != 0:\n",
    "        n += 1\n",
    "    x = np.linspace(a, b, n+1)\n",
    "    y = f(x)\n",
    "    return (b - a) / (3 * n) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])\n",
    "\n",
    "# Example usage\n",
    "def f(x):\n",
    "    return np.sin(x)\n",
    "\n",
    "a, b = 0, np.pi\n",
    "n = 100\n",
    "\n",
    "print(f\"Trapezoidal Rule: {trapezoidal_rule(f, a, b, n)}\")\n",
    "print(f\"Simpson's Rule: {simpson_rule(f, a, b, n)}\")\n",
    "print(f\"Actual value: {-np.cos(np.pi) + np.cos(0)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "3a3a55b3-430f-49a4-81ba-4444b23be096",
   "metadata": {},
   "outputs": [
    {
     "data": 
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Simple Linear Regression: y = 2.20 + 0.60x\n",
      "\n",
      "Multiple Linear Regression:\n",
      "y = 1.00 + 0.00x1 + 1.00x2\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "# Simple Linear Regression\n",
    "x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)\n",
    "y = np.array([2, 4, 5, 4, 5])\n",
    "\n",
    "model = LinearRegression()\n",
    "model.fit(x, y)\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.scatter(x, y, color='blue')\n",
    "plt.plot(x, model.predict(x), color='red')\n",
    "plt.title('Simple Linear Regression')\n",
    "plt.xlabel('x')\n",
    "plt.ylabel('y')\n",
    "plt.show()\n",
    "\n",
    "print(f\"Simple Linear Regression: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x\")\n",
    "\n",
    "# Multiple Linear Regression\n",
    "X = np.array([[1, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4], [4, 5]])\n",
    "Y = np.array([2, 3, 3, 4, 4, 5, 5, 6])\n",
    "\n",
    "model = LinearRegression()\n",
    "model.fit(X, Y)\n",
    "\n",
    "print(\"\\nMultiple Linear Regression:\")\n",
    "print(f\"y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x1 + {model.coef_[1]:.2f}x2\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "8ffef790-8822-4fa9-aa65-2025011e33ff",
   "metadata": {},
   "outputs": [
    {
     "data": 
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def numerical_derivative(f, x, h=1e-5):\n",
    "    return (f(x + h) - f(x - h)) / (2 * h)\n",
    "\n",
    "# Example function\n",
    "def f(x):\n",
    "    return x**2 * np.sin(x)\n",
    "\n",
    "# Calculate derivative\n",
    "x = np.linspace(0, 2*np.pi, 100)\n",
    "y = f(x)\n",
    "dy_dx = numerical_derivative(f, x)\n",
    "\n",
    "# Plot\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(x, y, label='f(x)')\n",
    "plt.plot(x, dy_dx, label=\"f'(x)\")\n",
    "plt.legend()\n",
    "plt.title('Function and its Derivative')\n",
    "plt.xlabel('x')\n",
    "plt.ylabel('y')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "036acee3-648c-482f-9824-ab795c8721a8",
   "metadata": {},
   "outputs": [
    {
     "data": 
   
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def polynomial_fit(x, y, degree):\n",
    "    coeffs = np.polyfit(x, y, degree)\n",
    "    p = np.poly1d(coeffs)\n",
    "    return p\n",
    "\n",
    "# Generate some noisy data\n",
    "x = np.linspace(0, 10, 100)\n",
    "y = 2 * x**2 - 5 * x + 3 + np.random.normal(0, 10, 100)\n",
    "\n",
    "# Fit polynomials of different degrees\n",
    "p2 = polynomial_fit(x, y, 2)\n",
    "p5 = polynomial_fit(x, y, 5)\n",
    "\n",
    "# Plot results\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.scatter(x, y, label='Data')\n",
    "plt.plot(x, p2(x), label='2nd degree fit')\n",
    "plt.plot(x, p5(x), label='5th degree fit')\n",
    "plt.legend()\n",
    "plt.title('Polynomial Curve Fitting')\n",
    "plt.xlabel('x')\n",
    "plt.ylabel('y')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "96a3ded4-63aa-488e-8676-528cf89cdbbd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 1: x = 1.666667, f(x) = -0.888889\n",
      "Iteration 2: x = 1.909091, f(x) = -0.264463\n",
      "Iteration 3: x = 1.976744, f(x) = -0.069227\n",
      "Final approximation: x = 1.976744\n"
     ]
    }
   ],
   "source": [
    "def f(x):\n",
    "    return x**2 - x - 2\n",
    "\n",
    "def regula_falsi(a, b, iterations):\n",
    "    for i in range(iterations):\n",
    "        fa = f(a)\n",
    "        fb = f(b)\n",
    "        x = (a * fb - b * fa) / (fb - fa)\n",
    "        fx = f(x)\n",
    "        \n",
    "        print(f\"Iteration {i+1}: x = {x:.6f}, f(x) = {fx:.6f}\")\n",
    "        \n",
    "        if fx * fa < 0:\n",
    "            b = x\n",
    "        else:\n",
    "            a = x\n",
    "    \n",
    "    return x\n",
    "\n",
    "# Initial guesses\n",
    "a, b = 1, 3\n",
    "\n",
    "# Perform 3 iterations\n",
    "result = regula_falsi(a, b, 3)\n",
    "print(f\"Final approximation: x = {result:.6f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4137dca8-edc0-4535-9185-4b4003b5b2ed",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
