def linear_1p(fname, x_title='x-title', y_title='y-title', 
              graph_title='Data with Best Fit', data_label='Measured Data'):
    """
    Fits a linear y=mx model to the data in the specified CSV file, producing a best-fit and residuals plots
    as well as fit parameters and goodness of fit measures. This is an analytic solution.

    Parameters:
    - fname (str): The filename of the CSV file containing the data.
    - x_title (str): x-axis title (optional).
    - y_title (str): y-axis title (optional).
    - graph_title (str): Graph title for upper plot (optional).
    - data_label (str): Legend label for the data set (optional).

    Returns:
    - slope (float): The slope of the linear fit.
    - slope_uncertainty (float): The uncertainty in the slope.
    - chi2 (float): The chi-squared value of the fit.
    - dof (int): Degrees of freedom.
    """

    import math
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load data from the CSV file
    data = np.loadtxt(fname, delimiter=',', comments='#', usecols=(0, 1, 2, 3))

    # Extract data columns (note that dx is not used)
    x = data[:, 0]
    y = data[:, 2]
    y_sigma = data[:, 3]

    # Define the fit function
    def fit_function(x, m):
        return m * x

    # Calculate the best fit slope analytically (1-parameter solution)
    m = sum(x * y / y_sigma**2) / sum((x / y_sigma)**2)

    # Calculate uncertainty of the best fit slope
    m_sigma = math.sqrt(1 / sum((x / y_sigma)**2))

    print("Slope:", m, "±", m_sigma)

    # Calculate and print the chi-squared, degrees of freedom, and weighted chi-squared
    def chi_square(param, x, y, sigma):
        return np.sum((y - fit_function(x, param))**2 / sigma**2)

    dof = len(x) - 1
    chi2 = chi_square(m, x, y, y_sigma)/dof
    
    print("\nReduced chi-squared: {}".format(chi2))
    
    # Plot the data and the fit curve, plus a plot of residuals
    x_fitfunc = np.linspace(min(x), max(x), 500)
    y_fitfunc = fit_function(x_fitfunc, m)

    fig = plt.figure(figsize=(7, 10))

    # Plot the data and best fit
    plot1 = fig.add_subplot(2, 1, 1)
    plot1.errorbar(x, y, yerr=y_sigma, marker='.', linestyle='', label=data_label)
    plot1.plot(x_fitfunc, y_fitfunc, marker="", linestyle="-", linewidth=2, color="r", label="1-Parameter Linear Fit")
    plot1.set_xlabel(x_title)
    plot1.set_ylabel(y_title)
    plot1.set_title(graph_title)
    plot1.legend(loc='best', numpoints=1)

    # Calculate and plot residuals
    y_fit = fit_function(x, m)
    residual = y - y_fit
    norm_residual = residual / y_sigma

    # Plot residuals
    plot2 = fig.add_subplot(212)
    plot2.errorbar(x, residual, yerr=y_sigma, marker='.', linestyle='', label="residual (y-y_fit)")
    plot2.hlines(0, np.min(x), np.max(x), lw=2, alpha=0.8)
    plot2.set_xlabel(x_title)
    plot2.set_ylabel('y-y_fit')
    plot2.legend(loc='best', numpoints=1)

    plt.show()

    return m, m_sigma, chi2



def linear_2p(fname, x_title='x-title', y_title='y-title', 
              graph_title='Data with Best Fit', data_label='Measured Data'):
    """
    Fits a linear y=mx+b model to the data in the specified CSV file, producing a best-fit and residuals plots
    as well as fit parameters and goodness of fit measures. This is an analytic solution.

    Parameters:
    - fname (str): The filename of the CSV file containing the data.
    - x_title (str): x-axis title (optional).
    - y_title (str): y-axis title (optional).
    - graph_title (str): Graph title for upper plot (optional).
    - data_label (str): Legend label for the data set (optional).
    
    Returns:
    - slope (float): The slope of the linear fit.
    - slope_uncertainty (float): The uncertainty in the slope.
    - chi2 (float): The chi-squared value of the fit.
    - dof (int): Degrees of freedom.
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    
    def fit_function(x, m, b):
        return m * x + b

    # Load data from the CSV file
    data = np.loadtxt(fname, delimiter=',', comments='#', usecols=(0, 1, 2, 3))

    # Extract data columns (note that dx is not used)
    x = data[:, 0]
    y = data[:, 2]
    y_sigma = data[:, 3]
    x_fitfunc = np.linspace(min(x), max(x), 500)

    Delta = np.sum(1 / y_sigma**2) * np.sum((x / y_sigma)**2) - (np.sum(x / y_sigma**2)**2)
    m = (np.sum(1 / y_sigma**2) * np.sum(x * y / y_sigma**2) - np.sum(x / y_sigma**2) * np.sum(y / y_sigma**2)) / Delta
    m_sigma = math.sqrt(np.sum(1 / y_sigma**2) / Delta)

    b = (np.sum((x / y_sigma)**2) * np.sum(y / y_sigma**2) - np.sum(x / y_sigma**2) * np.sum(x * y / y_sigma**2)) / Delta
    b_sigma = math.sqrt(np.sum((x / y_sigma)**2) / Delta)

    print("Slope:", m, "±", m_sigma)
    print("\ny-intercept", b, "±", b_sigma)
    
    dof = len(x) - 2
    chi2 = np.sum((y - fit_function(x, m, b))**2 / y_sigma**2)/dof
    
    print("\nReduced chi-squared: {}".format(chi2))

    y_fitfunc = fit_function(x_fitfunc, m, b)

    fig = plt.figure(figsize=(7, 10))

    plot1 = fig.add_subplot(2, 1, 1)
    plot1.errorbar(x, y, yerr=y_sigma, marker='.', linestyle='', label=data_label)
    plot1.plot(x_fitfunc, y_fitfunc, marker="", linestyle="-", linewidth=2, color="r", label=" fit")
    plot1.set_xlabel(x_title)
    plot1.set_ylabel(y_title)
    plot1.set_title(graph_title)
    plot1.legend(loc='best', numpoints=1)

    y_fit = fit_function(x, m, b)
    residual = y - y_fit
    normresidual = residual / y_sigma

    plot2 = fig.add_subplot(212)
    plot2.errorbar(x, residual, yerr=y_sigma, marker='.', linestyle='', label="residual (y-y_fit)")
    plot2.hlines(0, np.min(x), np.max(x), lw=2, alpha=0.8)
    plot2.set_xlabel(x_title)
    plot2.set_ylabel('y-y_fit')
    plot2.legend(loc='best', numpoints=1)

    plt.show()
    
    return m, m_sigma, b, b_sigma, chi2

