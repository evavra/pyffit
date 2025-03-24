import os 
import time 
import h5py
import numba 
import numpy as np
# from okada_wrapper
import cutde.halfspace as HS
from scipy.interpolate import interp1d
from itertools import combinations
from pyffit.utilities import rotate
import matplotlib.pyplot as plt

# ---------- Classes ----------
class TriFault:
    """
    Finite fault model specified by a set of rectangular slip patches.

    ATTRIBUTES:
    patches - collection of Patch objects that compose the fault

    """

    def __init__(self, mesh_file, triangle_file, slip=[], slip_components=[0, 1, 2], poisson_ratio=0.25, shear_modulus=30e9, mu=0, eta=0, reg_matrix_dir='.', 
                 verbose=False, trace_inc=0.01, N_data=True):
        """
        Instantiate TriFault object
        """

        # Geometry
        self.mesh         = np.loadtxt(mesh_file, delimiter=',', dtype=float)
        self.triangles    = np.loadtxt(triangle_file, delimiter=',', dtype=int) # CONFIRM MESH USES PYTHON INDEXING
        self.patches      = self.mesh[self.triangles]
        self.trace        = self.mesh[self.mesh[:, 2] == 0][:, :2]

        # Physical parameters
        self.slip_components = slip_components
        self.poisson_ratio   = poisson_ratio
        self.shear_modulus   = shear_modulus
        self.slip            = slip
        self.moment          = np.nan
        self.magnitude       = np.nan

        # Regularization
        self.mu = mu
        self.eta = eta

        # Other
        self.verbose = verbose
        self.N_data = N_data

        # Interpolate trace to specfied increment
        if trace_inc > 0:
            interp  = interp1d(self.trace[:, 0], self.trace[:, 1])
            x_trace = np.linspace(self.trace[:, 0].min(), self.trace[:, 0].max(), int((self.trace[:, 0].max() - self.trace[:, 0].min())/trace_inc) + 1)
            y_trace = interp(x_trace)
            self.trace_interp = np.array([x_trace, y_trace]).T

        # Compute moment/magnitude
        if len(slip) == len(self.patches):
            self.moment, self.magnitude = get_moment_magnitude(self.mesh, self.triangles, self.slip, shear_modulus=self.shear_modulus)

        # Check to see if regularization file exists
        reg_file = f'{reg_matrix_dir}/regularization.h5'

        if os.path.exists(reg_file):
            # Load file
            f = h5py.File(reg_file, 'r')
            R = f['R'][()]
            E = f['E'][()]
            f.close()

            # If wrong size, redo
            if R.shape[0] != self.triangles.shape[0]:
                R, E = get_regularization_matrices(self.mesh, self.triangles, reg_file)

        # Otherwise, make matrices and save to file
        else:
            R, E = get_regularization_matrices(self.mesh, self.triangles, reg_file)

            # Load file
            f = h5py.File(reg_file, 'w')
            f.create_dataset('R', data=R)
            f.create_dataset('E', data=E)
            f.close()
        
        self.smoothing_matrix = R
        self.edge_slip_matrix = E


    def greens_functions(self, x, y, z=[], look=[], disp_components=[0, 1, 2], slip_components=[0, 1, 2], rotation=np.nan, squeeze=True):
        """
        Compute fault Green's functions for given data coordinates and fault model.
    
        INPUT:
        x, y            - coordinates for Green's function locations
        disp_components -  slip components to use [0 for east, 1 for north, 2 for vertical] 
                           if rotating into strike coordiantes, it is [0 for fault-perpendicular, 1 for fault-parallel, 2 for vertical]
        slip_components -  slip components to use [0 for strike-slip, 1 for dip-slip, 2 for opening]

        OUTPUT:
        GF - array containing Green's functions for each fault element for each data point 
             Dimensions: N_OBS_PTS, 3 (E/N/Z), N_SRC_TRIS, 3 (srike-slip/dip-slip/opening)]
        """

        # Start timer
        start = time.time()

        if len(z) == 0 :
            z = np.zeros_like(x)

        # Prep coordinates and generate Green's functions
        pts = np.array([x, y, z]).reshape((3, -1)).T.copy() # Convert observation coordinates into row-matrix
        GF  = HS.disp_matrix(obs_pts=pts, tris=self.patches, nu=self.poisson_ratio) # (N_OBS_PTS, 3, N_SRC_TRIS, 3)

        # Select which slip components to use
        if len(slip_components) < 3:
            print(f'Using slip components {slip_components}')
            GF = GF[:, :, :, slip_components] 

        # If specified, project into LOS
        if len(look) == len(x):
            print('Projecting to LOS...')
            GF = self.proj_to_LOS(GF, look)

        elif (len(look) != 0):
            print('Error! Look vectors are wrong dimension')
            print(f'GF:   {GF.shape}')
            print(f'Look: {look.shape}')
            return
        
        # If not, rotate Greens functions and/or select displacement components to retur
        else:
            # Rotate horizontal components according to specified angle
            if (~np.isnan(rotation)) :
                print(f"Projecting Green's functions to {rotation} degrees")
                GF[:, 0, :, :], GF[:, 1, :, :] = rotate(GF[:, 0, :, :], GF[:, 1, :, :], np.deg2rad(rotation))

            # Select which displacement components to use
            if len(disp_components) < 3:
                print(f'Using displacement components {disp_components}')
                GF = GF[:, disp_components, :, :] 

        # Squeeze
        if squeeze:
            print('Squeezing')
            GF = np.squeeze(GF)

        # Stop timer
        end = time.time() - start

        # Display info
        if self.verbose:
            # print(f"Green's function array size:       {GF.reshape((-1, self.triangles.size)).shape} {GF.size:.1e} elements")
            print(f"Green's function array size:       {GF.shape} {GF.size:.1e} elements")
            print(f"Green's function computation time: {end:.2f}")

        return GF


    def proj_to_LOS(self, GF, look):
        """
        Project fault Green's functions into direction specfied by input vectors

        INPUT:
        GF (n_obs, 3, n_patch, 3) - array of Green's functions (observations, disp. components, patches, slip components) 
        look (n_obs, 3)           - array of unit vector components

        OUTPUT:
        G_LOS (n_obs, n_patch, 3) - LOS Green's functions
        """ 

        start = time.time()

        GF_LOS = np.empty((GF.shape[0], GF.shape[2], GF.shape[3]))

        for i in range(GF.shape[0]):
            for j in range(GF.shape[2]):
                for k in range(GF.shape[3]):
                    GF_LOS[i, j, k] = np.dot(GF[i, :, j, k], look[i, :])
        
        end = time.time() - start

        if self.verbose:
            print(f"LOS Green's function array size:      {GF_LOS.shape} {GF_LOS.size:.1e} elements")
            print(f"LOS Green's function computation time: {end:.2f}")

        return GF_LOS
    

    def LOS_greens_functions(self, x, y, look):
        """
        Project fault Green's functions into direction specfied by input vectors

        INPUT:
        G (n_obs, 3, n_patch, 3) - array of Green's functions
        look  (n_obs, 3) - array of unit vector components

        OUTPUT:
        G_LOS (n_obs, n_patch, 3)
        """ 

        GF = self.greens_functions(x, y)

        n_comp = len(self.slip_components)

        start = time.time()

        GF_LOS = np.empty((GF.shape[0], GF.shape[2], n_comp))

        for i in range(GF.shape[0]):
            for j in range(GF.shape[2]):
                for k in range(n_comp):
                    GF_LOS[i, j, k] = np.dot(GF[i, :, j, k], look[i, :])
        
        end = time.time() - start

        if self.verbose:
            print(f"LOS Green's function array size:      {GF_LOS.shape} {GF_LOS.size:.1e} elements")
            print(f"LOS Green's function computation time: {end:.2f}")

        return GF_LOS


    def resolution_matrix(self, x, y, look=[], rotation=np.nan, disp_components=[0, 1, 2], slip_components=[0, 1, 2], smoothing=True, edge_slip=True, squeeze=False, mode='data'):
        """
        Compute data or model resolution matrix for a given set of observation coordinates and fault mesh.
        """

        # Compute Greens Functions
        if len(look) == 0:
            G = np.squeeze(self.greens_functions(x, y, look=look, rotation=rotation, disp_components=disp_components, slip_components=slip_components, squeeze=squeeze))

        # # Project to LOS
        # else:
        #     GF = np.squeeze(self.LOS_greens_functions(x, y, look))

        # Add regularization  
        if smoothing:
            G = np.vstack((G, self.mu*self.smoothing_matrix))  
        if edge_slip:
            G = np.vstack((G, self.eta*self.edge_slip_matrix))  

        # R = self.mu*self.smoothing_matrix
        # L = self.eta*self.edge_slip_matrix
        # G = np.vstack((GF, R, L))  
        # G = np.vstack((GF, self.mu*self.smoothing_matrix)) 
        # G = make_observation_matrix(GF, self.mu*self.smoothing_matrix)

        def plot_matrix(A, name):

            vlim = np.max(np.abs(A))
            fig, ax = plt.subplots(figsize=(10, 10 * A.shape[0]/A.shape[1]))
            ax.set_title(f'Dimensions:  {A.shape} ({A.size}) | Avg {A.max():.3f} {A.std():.3f} | Range {A.min():.3f} {A.max():.3f}')
            im = ax.imshow(A, vmin=-vlim, vmax=vlim, cmap='coolwarm', interpolation='none')
            plt.colorbar(im)
            plt.savefig(f'/Users/evavra/Software/pyffit/tests/NIF/Resolution_G_matrix/{name}_matrix_{A.shape[0]}x{A.shape[1]}.png', dpi=300)
            plt.close()

            return

        # Compute the generalized inverse of the Greens functions
        GtG     = G.T @ G
        # GtG_inv = np.linalg.solve(GtG, np.eye(len(GtG)))
        GtG_inv = np.linalg.pinv(GtG)
        G_g     = GtG_inv @ G.T

        # # Perform SVD
        # V, S, U = np.linalg.svd(G_g, full_matrices=True)
        # a = np.hstack((np.diag(S**-1), np.zeros((len(S), len(U) - len(S)))))
        # b = np.vstack((np.diag(S), np.zeros((len(U) - len(S), len(S)))))
        # N_SVD = (V @ a @ U.T) @ (U @ b @ V.T)

        # fig, ax = plt.subplots(figsize=(14, 8.2))
        # ax.set_title(f'Avg {S.mean():.3f} {S.std():.3f} | Range {S.min():.3f} {S.max():.3f}')
        # ax.plot(S)
        # plt.show()

        # plot_matrix(GF, 'GF')
        # plot_matrix(R, 'R')
        # plot_matrix(L, 'L')
        # plot_matrix(G_g, 'G_g')
        # plot_matrix(N_SVD, 'N_SVD')

        if mode =='model':
            N = G_g @ G

        # Data resolution matrix
        else:
            # Compute resolution matrix
            N = G @ G_g

            if self.N_data:
                N = N[:len(x), :len(x)] # ignore rows corresponding to regularization terms

        return N


    def model_resolution_matrix(self, x, y, look, mode='data'):
        """
        Compute model resolution matrix for a given set of observation coordinates and fault mesh.
        """

        # Compute Greens Functions

        # Project to LOS
        GF_LOS = np.squeeze(self.LOS_greens_functions(x, y, look))

        # Add regularization  
        G = np.vstack((GF_LOS, self.mu*self.smoothing_matrix, self.eta*self.edge_slip_matrix))      
        
        # Compute the generalized inverse of the Greens functions
        GtG     = G.T @ G
        GtG_inv = np.linalg.solve(GtG, np.eye(len(GtG)))
        G_g     = GtG_inv @ G.T


        if mode =='model':
            N = G_g @ G

        # Data resolution matrix
        else:
            # Compute resolution matrix
            N = G @ G_g

            if self.N_data:
                N = N[:len(look), :len(look)] # ignore rows corresponding to regularization terms

        return N
    

    def update_slip(self, slip):
        self.slip = slip
        self.moment, self.magnitude = get_moment_magnitude(self.mesh, self.triangles, self.slip, shear_modulus=self.shear_modulus)


def make_observation_matrix(G, R, t=1):
    """
    Form observation from Greens function matrix G and smoothing matrix R.
    """

    zeros_G = np.zeros_like(G)
    zeros_R = np.zeros_like(R)

    # Form rows
    data         = np.hstack((  G * t,       G, zeros_G))
    v_smooth     = np.hstack((      R, zeros_R, zeros_R))
    W_smooth     = np.hstack((zeros_R,       R, zeros_R))
    W_dot_smooth = np.hstack((zeros_R, zeros_R,       R))

    H            = np.vstack((data, v_smooth, W_smooth, W_dot_smooth))

    return H


class Fault:
    """
    Finite fault model specified by a set of rectangular slip patches.

    ATTRIBUTES:
    name    - name of fault
    patches - collection of Patch objects that compose the fault
    """

    def __init__(self, name):
        self.name = name
        self.patches = []
        self.n_patch = 0
        self.extent  = []
        self.trace   = Trace()
        self.strike  = np.nan


    def add_patch(self, x, y, z, slip=[0, 0, 0], avg_strike=np.nan, update_extent=True, update_trace=True):
        """
        Add a fault patch using Cartesian geometry
        """
        self.patches.append(Patch())
        self.patches[-1].add_cartesian_geometry(x, y, z, slip=slip, avg_strike=avg_strike)

        # Reset metadata related to patches
        self.n_patch = len(self.patches)

        if update_extent:
            patch_extents = np.empty((self.n_patch, 6))
            for i, patch in enumerate(self.patches):
                patch_extents[i, :] = patch.extent

            self.extent = [patch_extents[:, 0].min(), patch_extents[:, 1].max(), 
                           patch_extents[:, 2].min(), patch_extents[:, 3].max(),
                           patch_extents[:, 4].min(), patch_extents[:, 5].max()]

        # Update trace
        if update_trace:
            self.get_trace()

    def greens_functions(self, x, y, z, alpha, slip_modes=[0, 1, 2], components=[0, 1, 2], uniform_slip=False):
        """
        Aggregate all fault patch Green's functions into matrix G.

        INPUT:
        x, y, z    - observation locations
        alpha      - elastic medium constant (alpha = (lambda + mu)/(lambda + 2 * mu) = 1 - (Vs/Vp)**2)
        faults     - list containing Fault objects
        slip_modes - list containing slip modes to use (0=strike-slip, 1=dip slip, 2=opening)

        OUTPUT:
        G (3*n_obs, n_mode*n_patch) - Green's function design matrix for all patches in fault
        """
        n_obs  = len(x)
        n_mode = len(slip_modes)
        n_rows = len(components)

        if uniform_slip == False:
            G = np.empty((n_rows*n_obs, n_mode*self.n_patch))
            for j, patch in enumerate(self.patches):
                G[:, n_mode*j:n_mode*(j+1)] = patch.greens_functions(x, y, z, alpha, slip_modes=slip_modes, components=components)
        
        elif uniform_slip == True:
            G = np.zeros((n_rows*n_obs, n_mode))
            for j, patch in enumerate(self.patches):
                G += patch.greens_functions(x, y, z, alpha, slip_modes=slip_modes, components=components)
        return G

    def disp(self, x, y, z, alpha, slip='self', components=[0, 1, 2]):
        """
        Calculate displacements at coordinates (x, y, z) due to slip on fault.

        INPUT:
        x, y, z - (n,) arrays of observation coordinates
        alpha   - constituitive parameter
        slip    - (3,) array-like containing strike-slip/dip-slip/opening slip components or 'self' to
                  use existing patch slip value (default)
        components   - toggle easting/northing/vertical displacement components (0=east, 1=north, 2=up)

        OUTPUT:
        U       - (n, 3) array containing displacement vector components in the x/y/z directions
        """
        n_comp = len(components)

        # Compute displacements
        U = np.zeros((len(x), n_comp))

        for patch in self.patches:
            U += patch.disp(x, y, z, alpha, patch.slip, components=components)

        return U

    def get_trace(self,):
        """
        Extract x/y coordinates of the shallowest fault edge.
        """

        x = []
        y = []
        z = []

        for patch in self.patches:
            x.extend(patch.x)
            y.extend(patch.y)
            z.extend(patch.z)

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)

        x = x[z == z.min()]
        y = y[z == z.min()]
        z = z[z == z.min()]

        self.trace.x = x
        self.trace.y = y
        self.trace.z = z

        return x, y, z

    def get_average_strike(self,):
        """
        Extract average fault strike via 1D polynomial fit
        """

        p = np.polyfit(self.trace.x, self.trace.y, 1)

        return np.rad2deg(np.arctan(1/p[0])) + 180

    def get_length(self,):
        """
        Get fault length based on the fault trace
        """

        return sum([np.sqrt((self.trace.x[i + 1] - self.trace.x[i])**2 + (self.trace.y[i + 1] - self.trace.y[i])**2) for i in range(len(self.trace.x) - 1)]) 


class Patch:
    """
    Rectangular fault patch object with geometric and slip attributes.

    Can specify geometry using two schemes:

    1) Cartesian coordinates: x, y and z-coordinates of fault patch vertices
        Arguments:
            x - array of x-coordinates [x_a, x_b, x_c, x_d] 
            y - array of y-coordinates [y_a, y_b, y_c, y_d] 
            z - array of z-coordinates [z_a, z_b, z_c, z_d] 

    2) Patch location and dimensions: length, width, and dip along with "leading" vertex 
        Arguments:
            vertex - cartesian coordinates of "up-strike, up-dip" vertex [x_a, y_a].
            strike - fault patch strike  (deg)
            dip    - fault patch dip     (deg)
            l      - along-strike length (m)
            d      - along-dip length    (m)

    In both cases, the index refers to the a specific vertex
        a) up-strike,   up-dip
        b) down-strike, up-dip
        c) down-strike, down-dip
        d) up-strike,   down-dip

    slip - slip vector (unitary) [strike-slip, dip-slip, opening/closing]
    """

    def __init__(self):
        self.slip         = [0, 0, 0]
        self.x            = 0
        self.y            = 0
        self.z            = 0
        self.node         = []
        self.origin       = []
        self.strike       = 0
        self.strike_r     = 0
        self.dip          = 0            
        self.dip_r        = 0            
        self.l            = 0
        self.d            = 0
        self.strike_width = []
        self.dip_width    = []
        self.extent       = []


    def add_cartesian_geometry(self, x, y, z, slip=[0, 0, 0], avg_strike=np.nan):
        """
        Add patch geometry via cartesian coordinates (usually UTM).
        """

        # Add cartesian vertex coordinates
        self.x = x
        self.y = y
        self.z = z
        self.slip = slip

        # Compute additional attributes
        dx_l = x[0] - x[1]
        dy_l = y[0] - y[1]
        dx_d = x[0] - x[3]
        dy_d = y[0] - y[3]
        dz_d = z[0] - z[3]


        self.node         = np.array([x[0], y[0], z[0]])                     # Leading along-strike vertex (a)
        self.l            = np.sqrt(dx_l**2 + dy_l**2)                       # Strike width
        self.d            = np.sqrt(dx_d**2 + dy_d**2 + dz_d**2)             # Dip width
        self.strike       = get_strike(dx_l, dy_l)                           # Strike
        self.strike_r     = self.strike * (np.pi/180)                        # Strike
        self.dip_r        = np.arcsin(dz_d/self.d)                           # Dip in radians
        self.dip          = self.dip_r * (180/np.pi)                              # Dip in degrees
        self.origin       = [self.node[0] + self.l*np.sin(-self.strike_r)/2, # Mid-point of upper edge
                             self.node[1] - self.l*np.cos(-self.strike_r)/2,
                             z[0]] 
        self.strike_width = [-self.l/2, self.l/2]                            # along-strike extent in patch reference system
        self.dip_width    = [0, self.d]                                      # down dip extent in patch reference system
        self.extent       = [np.min(self.x), np.max(self.x),                 # Spatial extent of patch
                             np.min(self.y), np.max(self.y),
                             np.min(self.z), np.max(self.z),]

        # Rectify fault strike ambiguity by specifying preferred average strike
        if ~np.isnan(avg_strike):
            d_strike = self.strike + avg_strike

            if abs(d_strike) >= 180:
                new_strike = self.strike + d_strike - (d_strike % 180)
                self.strike = new_strike % 360

        # print(self.strike)


    def add_self_geometry(self, origin, strike, dip, l, d, slip=[0, 0, 0]):
        """
        Add patch geometry from patch origin, dimesions, and angles.
        """

        # Add base atrributes specified on instantiation.
        self.origin   = origin
        self.strike   = strike
        self.dip      = dip            
        self.strike_r = strike * (np.pi/180) 
        self.dip_r    = dip * (np.pi/180)   
        self.l        = l
        self.d        = d
        self.slip     = slip

        # Compute additional attributes
        x_r = np.array([0, 0, d * np.cos(self.dip_r) * dip/abs(dip), d * np.cos(self.dip_r) * dip/abs(dip)])
        y_r = np.array([l/2, -l/2, -l/2, l/2])
        x, y = rotate(x_r, y_r, -self.strike_r)


        self.strike_width  = [l/2, -l/2]
        # self.dip_width     = [d, 0]
        self.dip_width     = [-d, 0]
        self.x             = x + origin[0]
        self.y             = y + origin[1] 
        # self.z             = [0, 0, d * np.sin(self.dip_r), d * np.sin(self.dip_r)]
        self.z             = [origin[2], origin[2], origin[2] + d * abs(np.sin(self.dip_r)), origin[2] + d * abs(np.sin(self.dip_r))]
        self.node          = [self.x[0], self.y[0], self.z[0]]
        self.extent        = [np.min(self.x), np.max(self.x),
                              np.min(self.y), np.max(self.y),
                              np.min(self.z), np.max(self.z),]
 

    def poly(self, kind='edges', mode='3d', **kwargs):
        """
        Get polygon object for visualization purposes.

        kind = 'edges' for LineCollection or 'faces' for Poly3DCollection.
        mode = '2d' or '3d'
        """
        x = self.x
        y = self.y
        z = self.z

        if kind == 'edges':
            xl = np.append(x, x[0])
            yl = np.append(y, y[0])
            zl = np.append(z, z[0])

            if mode == '3d':
                return Line3DCollection([list(zip(xl, yl, zl))], **kwargs)
            else:
                return LineCollection([list(zip(xl, yl))], **kwargs)

        elif kind == 'face':
            if mode == '3d':
                return Poly3DCollection([list(zip(x, y, z))], **kwargs)
            else:
                return LineCollection([list(zip(x, y))], **kwargs)
        else:
            print("Must specify kind as 'edges' or 'faces'!")


    def disp(self, x, y, z, alpha, slip, components=[0, 1, 2]):
        """
        Calculate displacements at coordinates (x, y, z) due to slip on the fault patch.

        INPUT:
        x, y, z - (n,) arrays of observation coordinates
        slip    - (3,) array-like containing strike-slip/dip-slip/opening slip components or False to
                  use existing patch slip value (default)

        OUTPUT:
        U       - (n, 3) array containing displacement vector components in the x/y/z directions
        """

        n_obs = len(x)
        U = np.zeros((n_obs, len(components)))

        # Loop over each grid location
        for i in range(n_obs):
            # Rotate observation points from geographic coordinates to fault coordinates
            x_r, y_r = rotate(x[i] - self.origin[0], y[i] - self.origin[1], -np.pi/2 + self.strike_r) 
            
            # print(alpha, [x_r, y_r, z], self.origin[2], self.dip, self.strike_width, self.dip_width, slip)

            # Compute displacements due to fault
            # print(alpha, [x_r, y_r, z], self.origin[2], self.dip, self.strike_width, self.dip_width, slip)
            success, u, grad_u = dc3dwrapper(alpha, [x_r, y_r, z], self.origin[2], self.dip, self.strike_width, self.dip_width, slip)
            # assert(success == 0)
            
            # Account for singularities
            if success!= 0:
                print('DC3D error: IRET =', success)
                print(self.x)
                print(self.y)
                print(self.z)
                print(self.strike)
                print(self.dip)
                print(self.l)
                print(self.d)

                u = np.array([-np.inf, -np.inf, -np.inf])

            else:
                # Rotate displacements from fault coordinates to geographic coordinates
                u[:2] = rotate(u[0], u[1], np.pi/2 - self.strike_r)

            # Export results
            U[i, :] = u[components]

        return U


    def greens_functions(self, x, y, z, alpha, slip_modes=[0, 1, 2], components=[0, 1, 2]):
        """
        Get fault patch Green's functions which correspond to observation points at (x, y, z).
        unit_slip - list-like object selecting which slip components to use [strike-slip, dip-slip, opening] 
        
        INPUT:
        x, y, z    - observation locations
        alpha      - elastic medium constant (alpha = (lambda + mu)/(lambda + 2 * mu) = 1 - (Vs/Vp)**2)
        faults     - list containing Fault objects
        slip_modes - list containing slip modes to use (0=strike-slip, 1=dip slip, 2=opening)

        OUTPUT:
        g (3*n_obs, n_mode) - Green's function matrix where each row corresponds to a displacment component
                              at an observation point and the rows correspond to slip mode.
        """
        n_obs     = len(x)
        n_mode    = len(slip_modes)
        n_comp    = len(components)
        unit_slip = np.eye(3) # Rows correspond to unit slip vectors for strike-slip, dip slip, and opening modes
        g         = np.empty((n_comp*n_obs, n_mode))

        for i in range(n_obs): # Rows of g correspond to each displacment component at (x, y, z)
            for j, mode in enumerate(slip_modes): # Columns of g correpond to the contribution from each slip mode
                g[n_comp*i:n_comp*(i + 1), j] = self.disp([x[i]], [y[i]], [z[i]], alpha, unit_slip[mode, :], components=components)

        return g


class Trace:
    """
    Surface or uppermost edge of finite fault model

    ATTRIBUTES:
    x - x-coordinates of trace
    y - y-coordinates of trace
    z - z-coordinates of trace
    """

    def __init__(self):
        self.x = []
        self.y = []
        self.y = []


# ---------- Methods ----------
@numba.jit(nopython=True)
def rotate(x, y, theta): 
    """
    Perform 2D rotation about origin.
    """

    x_r = np.cos(theta) * x - np.sin(theta) * y
    y_r = np.sin(theta) * x + np.cos(theta) * y

    return x_r, y_r


def get_strike(dx, dy):
    """
    Compute fault patch strike from the component differences between
    leading and trailing upper verticies (a & b).
    """
    try:
        # Calculate strike 
        strike = np.arctan(dx/dy) * (180/np.pi)

        # Account for non-uniqueness in tangent function
        if dy < 0: # Lower quadrants
            strike += 180
        if (dy > 0) & (dx < 0): # Upper-right quadrant
            strike += 360

    # Account for singularities when fault strikes along x-axis
    except ZeroDivisionError as e:
        if dy == 0:
            if dx > 0:
                strike = 90
            else:
                strike = 270
    return strike


def make_simple_FFM(fault_dict, avg_strike=np.nan):
    """
    Generate vertical finite fault model from mapped (x, y) cooridinates and depth range.

    INPUT:
    fault_dict - dictionary with keys ['Name', UTMx', 'UTMy', 'z', 'slip']

    OUTPUT:
    fault - fault object (see pyfft.finite_fault for class descriptions)
    """

    # Get information from mapped fault dictionary
    name    = fault_dict['Name']    
    x       = fault_dict['UTMx']
    y       = fault_dict['UTMy']
    z       = fault_dict['z']
    slip    = fault_dict['slip']
    n_patch = len(x) - 1

    # Form vertex arrays (along-strike, down-dip)
    X = np.vstack((x, x)).T
    Y = np.vstack((y, y)).T
    Z = np.ones((n_patch + 1, 2)) * z
    
    # Assign patches to fault
    fault = Fault(name)

    for i in range(n_patch):
        # Parse coordinates
        x_p = np.array([X[i, 0], X[i + 1, 0], X[i + 1, 1], X[i, 1]])
        y_p = np.array([Y[i, 0], Y[i + 1, 0], Y[i + 1, 1], Y[i, 1]])
        z_p = np.array([Z[i, 0], Z[i + 1, 0], Z[i + 1, 1], Z[i, 1]])

        # Add patch to list
        fault.add_patch(x_p, y_p, z_p, slip, avg_strike=avg_strike)

    return fault


def get_fault_greens_functions(x, y, z, mesh, triangles, nu=0.25, verbose=True):
    """
    Compute fault Green's functions for given data coordinates and fault model.

    INPUT:


    OUTPUT:
    GF - array containing Green's functions for each fault element for each data point 
         Dimensions: N_OBS_PTS, 3 (E/N/Z), N_SRC_TRIS, 3 (srike-slip/dip-slip/opening)

    """

    # Start timer
    start = time.time()

    # Prep coordinates and generate Green's functions
    pts = np.array([x, y, z]).reshape((3, -1)).T.copy() # Convert observation coordinates into row-matrix
    GF  = HS.disp_matrix(obs_pts=pts, tris=mesh[triangles], nu=nu)    # (N_OBS_PTS, 3, N_SRC_TRIS, 3)

    # Stop timer
    end = time.time() - start

    # Display info
    if verbose:
        print(f"Green's function array size:      {GF.reshape((-1, triangles.size)).shape} {GF.size:.1e} elements")
        print(f"Green's function computation time: {end:.2f}")

    return GF


def get_fault_displacements(x, y, z, mesh, triangles, slip, nu=0.25, verbose=True):
    """
    Get surface displacements for given fault mesh and slip distribution.
    """

    # Start timer
    start      = time.time()

    # Prepare coordinates
    pts = np.array([x, y, z]).reshape((3, -1)).T.copy() 

    # Compute displacements
    disp = HS.disp_free(pts, mesh[triangles], slip, nu)

    # disp_grid  = disp.reshape((*data.shape, 3))
    # disp_model = np.array([np.dot(disp[i, :], look[i, :]) for i in range(len(disp))]) # project to LOS

    # Stop timer
    end        = time.time() - start

    if verbose:
        saveprint(f'Full displacements computation time: {end:.2f}')

    return disp


def get_full_disp(data, GF, slip, grid=False):
    """
    Compute full displacement field with original NaN values.
    """
    # Get nan locations
    i_nans = np.isnan(data.flatten())

    # Compute displacements
    disp = GF.dot(slip[:, 0].flatten())

    # Form output array
    disp_full = np.empty_like(data.flatten())
    disp_full[i_nans] = np.nan
    disp_full[~i_nans] = disp

    return disp_full


def proj_greens_functions(G, U, slip_components=[0, 1, 2], verbose=True):
    """
    Project fault Green's functions into direction specfied by input vectors

    INPUT:
    G (n_obs, 3, n_patch, 3) - array of Green's functions
    U  (n_obs, 3) - array of unit vector components

    OUTPUT:
    G_proj (n_obs, n_patch, 3)
    """ 

    n_comp = len(slip_components)

    start = time.time()

    G_proj = np.empty((G.shape[0], G.shape[2], n_comp))

    for i in range(G.shape[0]):
        for j in range(G.shape[2]):
            for k in range(n_comp):

                G_proj[i, j, k] = np.dot(G[i, :, j, k], U[i, :])
    
    end = time.time() - start

    if verbose:
        print(f"LOS Green's function array size:      {G_proj.shape} {G_proj.size:.1e} elements")
        print(f"LOS Green's function computation time: {end:.2f}")

    return G_proj


def get_fault_info(mesh, triangles, verbose=True):
    """
    Get information about fault mesh construction.

    INPUT:
    mesh (m, 3)       - x/y/z coordinates of mesh vertices
    triangles (mn, 3) - indicies of vertices for each nth triangular element

    OUTPUT:
    fault_info - dictionary containing the folowing attributes:
                n_vertex          - number of vertices
                n_patch           - number of patches
                depths            - depths of each layer
                layer_thicknesses - thicknesses of each layer
                trace             - x/y/z/ coordinates of fault surface trace
                n_top_patch       - number of surface patches
                l_top_patch       - along-strike length of surface patches
    """

    n_vertex = len(mesh)
    n_patch  = len(triangles)

    depths = abs(np.unique(mesh[:, 2])[::-1])
    depths_formatted  = ", ".join(f"{d:.2f}" for d in depths)
    layer_thicknesses = np.diff(depths)
    layer_thicknesses_formatted = ", ".join(f"{l:.2f}" for l in layer_thicknesses)

    trace       = mesh[mesh[:, 2] == 0]
    n_top_patch = len(trace) - 1
    l_top_patch = np.linalg.norm(trace[1:, :] - trace[:-1, :], axis=1)

    if verbose:
        print(f'# -------------------- Mesh info -------------------- ')
        print(f'Mesh vertices:           {n_vertex}')
        print(f'Mesh elements:           {n_patch}')
        print(f'Depths:                  {depths_formatted}')
        print(f'Layer thicknesses:       {layer_thicknesses_formatted}')
        print(f'Surface elements:        {n_top_patch}')
        print(f'Surface element lengths: {l_top_patch.min():.2f} - {l_top_patch.max():.2f}')

    return dict(n_vertex=n_vertex, n_patch=n_patch, depths=depths, layer_thicknesses=layer_thicknesses, trace=trace, n_top_patch=n_top_patch, l_top_patch=l_top_patch)


def get_moment_magnitude(mesh, triangles, slip_model, shear_modulus=30e9):
    """
    Compute moment and moment magnitude from a given fault slip model.
    """

    # Compute magnitude
    patch_area   = np.empty(len(slip_model))
    patch_slip   = np.empty(len(slip_model))
    patch_moment = np.empty(len(slip_model))

    for j in range(len(slip_model)):
        patch = mesh[triangles[j]]

        # Get patch area via cross product
        ab = patch[0, :] - patch[1, :]
        ac = patch[0, :] - patch[2, :]
        patch_area[j] = np.linalg.norm(np.cross(ab, ac), ord=2)/2 * 1e6 # convert from km^3 to m^3

        # Get slip magnitude
        patch_slip[j] = np.linalg.norm(slip_model[j, :], ord=2) * 1e-3 # convert from mm to m

        # Compute magnitude
        patch_moment[j] = shear_modulus * patch_area[j] * patch_slip[j] * 1e7 # convert from N-m to dyne-cm

    # Get moment and moment magnitude
    moment    = np.sum(patch_moment)
    magnitude = (2/3) * np.log10(moment) - 10.7

    return moment, magnitude


def get_regularization_matrices(mesh, triangles, filename):
    """
    Make regularziaiton matrices for given fault mesh.
    """

    print(f'Saving regularization matrices to {filename}')
    
    f = h5py.File(filename, 'w')

    # Get smoothness regularization matrix
    R = get_smoothing_matrix(mesh, triangles)

    # Get zero-slip regularization matrix
    E = get_edge_matrix(mesh, triangles, R)

    f.create_dataset('R', data=R)
    f.create_dataset('E', data=E)
    f.close()

    return R, E


def get_smoothing_matrix(mesh, triangles):
    """
    Form first-difference regularization matrix for triangular fault mesh.

    Each triangular element has three forward first-order differences:\

        ds_i/dr_i = (s_i - s_0)/||r_i - r_0||**2

    where s is the slip associate with each element, r are the element centroid coordinates, 
    index 0indicates the reference element, and index i indicates a neighboring element.


    INPUT:
    mesh (m, 3)      - x/y/z coordinates for mesh vertices
    triangles (n, 3) - row indices for mesh vertices corresponding to the nth element
    
    OUTPUT:
    R (n, n) - finite-difference matrix where each row encodes the finite-difference operator
               corresponding to the nth element.
    """

    R = np.zeros((len(triangles), len(triangles)))
    
    # Loop over each fault patch
    for i, tri in enumerate(triangles):
        pts = mesh[tri]

        # Get neighboring patches (shared edge)
        for cmb in combinations(tri, 2):

            # Check if each row contains the values
            for j, tri0 in enumerate(triangles):
                result = np.isin(cmb, tri0)

                # Neighbor triangle if two vertices are shared and not the original triangle
                if (sum(result) == 2) & (i != j):

                    # Nearest-neighbor only
                    R[i, j] -= 1
                    R[i, i] += 1
                    break

    return R


def get_edge_matrix(mesh, triangles, R):
    """
    Place zero-slip condition along bottom and side edges of fault.
    """

    E = np.zeros_like(R)

    # Smoothing-matrix method
    # Get number of finite differences for each element
    # n_diff = np.sum(np.abs(R) > 0, axis=1)

    # for i in range(len(R)):

        # Check if edge element by counting number of finite differences
        # 4 if fully surrounded by neighboring elements
        # 3 if one side is an edge
        # 2 if two sides are edges (corner element)

        # if n_diff[i] != 4:
        #     E[i, i] = n_diff[i]

        #     pts = mesh[triangles[i]][:, 2]
        #     print(pts)
            # If not at surface
            # if 0 in z_pts:
            #     print(i, z_pts)

            # if 0 not in z_pts:
                
                # E[i, i] = 1


    # Geometric method
    xmin = mesh[:, 0].min()
    xmax = mesh[:, 0].max()
    ymin = mesh[:, 1].min()
    ymax = mesh[:, 1].max()
    zmin = mesh[:, 2].min()
    zmax = mesh[:, 2].max()

    for i in range(len(R)):
        # Get vertices of each triangle
        pts = mesh[triangles[i]]

        # If any vertex is on the mesh edge (except for the top), add boundary constraint
        cond = ((xmin in pts[:, 0]) | (xmax in pts[:, 0]) | (ymin in pts[:, 1]) | (ymax in pts[:, 1]) | (zmin in pts[:, 2])) & (zmax not in pts[:, 2])
        # cond = (zmax not in pts[:, 2])

        if cond:
            E[i, i] = 1

    return E



