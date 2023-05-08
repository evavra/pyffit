import numpy as np
from pyffit.data import read_traces
from pyffit.finite_fault import make_simple_FFM

# Parameters
version        = 7
file_faults    = f'/Users/evavra/Projects/Turkey/Data/Model_faults_{version}.csv'
fault_names    = [
                  'Amanos',
                  'CAFZ',
                  'Çardak',
                  'EAF',
                  # 'Savrun',
                  # 'YGFZ',
                  # 'Pazarcık',
                  # 'Erkenek',
                  # 'Pütürge',
                  ]
EPSG           = '32637' # For Turkey
poisson_ratio  = 0.25
shear_modulus  = 30 * 10**9 # From Turcotte & Schubert
avg_strike     = 235 # approximate average fault strike to determine strike convention

# Set up elastic parameters
lmda  = 2 * shear_modulus * poisson_ratio / (1 - 2 * poisson_ratio)
alpha = (lmda + shear_modulus) / (lmda + 2 * shear_modulus)   

# Load faults
traces = read_traces(file_faults, 'QGIS', EPSG=EPSG)
faults = dict((name, traces[name]) for name in fault_names)

for name in faults:
    faults[name]['slip'] = [0, 0, 0]
    faults[name]['z']    = [0, 0]


def finite_fault_model(m, coords):
    """
    Generate and compute horizontal displacements for a network of finite faults.

    INPUT:
    m      (m+2, 2) - slip rates and locking depths corresponding to each fault in the model,
                      as well as east and west translation components
    coords (n, 3)   - x/y/z coordinates associated with model predictions

    OUTPUT:
    U      (2*n,)    - vector of model predicted east and north displacements
    """

    x, y, z = coords
    U = np.zeros((len(x), 2))

    for i_slip, name in enumerate(faults):
        i_lock = i_slip + len(faults)

        # Update slip and locking depth and generate fault model
        faults[name]['slip'][0] = m[i_slip]
        faults[name]['z'][0]    = m[i_lock]
        fault = make_simple_FFM(faults[name], avg_strike=avg_strike)
        # print(m[i_slip], m[i_lock])
        # print(faults[name]['slip'][0], faults[name]['z'][0])

        # Comppute displacements
        U += fault.disp(x, y, z, alpha, components=[0, 1])

    # Add translation vector to model output
    U[:, 0] += m[-2]
    U[:, 1] += m[-1]

    return U.flatten()