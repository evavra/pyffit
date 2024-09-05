import time 
import numba 
import numpy as np
# from okada_wrapper
import cutde.halfspace as HS


# ---------- Classes ----------
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
        Aggregate all fault patch Greens functions into matrix G.

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
            success, u, grad_u = dc3dwrapper(alpha, [x_r, y_r, z], self.origin[2], self.dip, self.strike_width, self.dip_width, slip)
            # assert(success == 0)
            
            # Account for singularities
            if success != 0:
                print('\n DC3D error: IRET =', success)
                # print(self.x)
                # print(self.y)
                # print(self.z)
                # print(self.strike)
                # print(self.dip)
                # print(self.l)
                # print(self.d)

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
        g (3*n_obs, n_mode) - Greens function matrix where each row corresponds to a displacment component
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
    Compute fault Greens functions for given data coordinates and fault model.

    INPUT:


    OUTPUT:
    GF - array containing Greens functions for each fault element for each data point 
         Dimensions: N_OBS_PTS, 3 (E/N/Z), N_SRC_TRIS, 3 (srike-slip/dip-slip/opening)

    """
    # Start timer
    start = time.time()

    # Prep coordinates and generate Greens functions
    pts = np.array([x, y, z]).reshape((3, -1)).T.copy() # Convert observation coordinates into row-matrix
    GF  = HS.disp_matrix(obs_pts=pts, tris=mesh[triangles], nu=nu) # (N_OBS_PTS, 3, N_SRC_TRIS, 3)

    # Stop timer
    end = time.time() - start

    # Display info
    if verbose:
        print(f'Greens function array size:      {GF.reshape((-1, triangles.size)).shape} {GF.size:.1e} elements')
        print(f'Greens function computation time: {end:.2f}')

    return GF


def get_fault_displacements(x, y, z, mesh, triangles, slip, nu=0.25, verbose=False):
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
        # print(f'Full displacements computation time: {end:.2f}')
        print(f'Full displacements computation time: {end:.2f}')

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


def proj_greens_functions(G, U, verbose=False):
    """
    Project fault Greens functions into direction specfied by input vectors.
        n_obs  - number of observations
        n_patch - number of fault patches
        n_mode  - number of slip modes

    INPUT:
    G (n_obs, 3, n_patch, n_mode) - array of Greens functions
    U  (n_obs, 3)                 - array of unit vector components

    OUTPUT:
    G_proj (n_obs, n_patch, n_mode) - array of LOS Greens functions
    """ 
    start = time.time()

    G_proj = np.empty((G.shape[0], G.shape[2], G.shape[3]))

    for i in range(G.shape[0]):
        for j in range(G.shape[2]):
            for k in range(G.shape[3]):

                G_proj[i, j, k] = np.dot(G[i, :, j, k], U[i, :])
    
    end = time.time() - start

    if verbose:
        print(f'LOS Greens function array size:      {G_proj.shape} {G_proj.size:.1e} elements')
        print(f'LOS Greens function computation time: {end:.2f}')

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
    depths_formatted = ", ".join(f"{d:.2f}" for d in depths)
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
