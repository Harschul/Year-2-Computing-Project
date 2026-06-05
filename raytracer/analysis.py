"""Analysis module."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from operator import itemgetter
from raytracer._utils.decorators import SaveOutput
from raytracer.rays import Ray, RayBundle
from raytracer.elements import SphericalRefraction, OutputPlane
from raytracer.lenses import PlanoConvex, BiConvex


def task8():
    """
    Task 8.

    In this function you should check your propagate_ray function properly
    finds the correct intercept and correctly refracts a ray. Don't forget
    to check that the correct values are appended to your Ray object.
    """
    positions = [[0, 0, 1], [0, 1, 1]]
    directions = [[0, 1, 1], [0, 1, 1]]

    if len(positions) != len(directions):
        raise RuntimeError("Please ensure a position and direction is defined for each ray")

    for pos, direc in zip(positions, directions):
        ray = Ray(pos, direc)

        sr = SphericalRefraction(
            z_0=10,
            aperture=50,
            curvature=0.01,
            n_1=1,
            n_2=1.5,
        )

        intercept = sr.intercept(ray)
        while intercept is not None:
            #print(intercept)
            sr.propagate_ray(ray)
            intercept = sr.intercept(ray)


@SaveOutput("task10")
def task10():
    """
    Task 10.

    In this function you should create Ray objects with the given initial positions.
    These rays should be propagated through the surface, up to the output plane.
    You should then plot the tracks of these rays.
    This function should return the matplotlib figure of the ray paths.

    Returns:
        Figure: the ray path plot.
    """

    positions = np.array([
        [0, 4, 0],
        [0, 1, 0],
        [0, 0.2, 0],
        [0, 0, 0],
        [0, -0.2, 0],
        [0, -1, 0],
        [0, -4, 0],
    ])
    directions = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ])

    if len(positions) != len(directions):
        raise RuntimeError("Please ensure a position and direction is defined for each ray")

    sr = SphericalRefraction(
         z_0=100,
         aperture=34,
         curvature=0.03,
         n_1=1,
         n_2=1.5,
        )

    op = OutputPlane(250)

    fig = plt.figure()
    plt.xlabel("z (mm)")
    plt.ylabel("y (mm)")

    for pos, direc in zip(positions, directions):
        ray = Ray(pos, direc)
        intercept = sr.intercept(ray)
        print(intercept)
        sr.propagate_ray(ray)
        op.propagate_ray(ray)
        z_values = np.array(ray.vertices())[:, 2]
        y_values = np.array(ray.vertices())[:, 1]
        plt.plot(z_values, y_values)

    return fig



@SaveOutput("task11", plot_output_indices=itemgetter(0))
def task11():
    """
    Task 11.

    In this function you should propagate the three given paraxial rays through the system
    to the output plane and the tracks of these rays should then be plotted.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for ray paths
    2. the calculated focal point.

    Returns:
        tuple[Figure, float]: the ray path plot and the focal point
    """

    positions = np.array([
        [0.1, 0.1, 0],
        [0, 0, 0],
        [-0.1, -0.1, 0],
    ])
    directions = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ])

    if len(positions) != len(directions):
        raise RuntimeError("Please ensure a position and direction is defined for each ray")

    sr = SphericalRefraction(
         z_0=100,
         aperture=34,
         curvature=0.03,
         n_1=1,
         n_2=1.5,
        )

    focal_point = sr.focal_point()
    op = OutputPlane(focal_point)

    fig = plt.figure()
    plt.xlabel("z (mm)")
    plt.ylabel("y (mm)")

    for pos, direc in zip(positions, directions):
        ray = Ray(pos, direc)
        intercept = sr.intercept(ray)
        print(intercept)
        sr.propagate_ray(ray)
        op.propagate_ray(ray)
        z_values = np.array(ray.vertices())[:, 2]
        y_values = np.array(ray.vertices())[:, 1]
        plt.plot(z_values, y_values)

    return fig, focal_point



@SaveOutput("task12")
def task12():
    """
    Task 12.

    In this function you should create a RayBunble and propagate it to the output plane
    before plotting the tracks of the rays.
    This function should return the matplotlib figure of the track plot.

    Returns:
        Figure: the track plot.
    """
    bundle = RayBundle()
    sr = SphericalRefraction(
         z_0=10,
         aperture=34,
         curvature=0.03,
         n_1=1,
         n_2=2,
        )

    focal_point = sr.focal_point()
    op = OutputPlane(focal_point)
    elements = [sr, op]
    bundle.propagate_bundle(elements)
    plot = bundle.track_plot()
    return plot


@SaveOutput("task13", plot_output_indices=itemgetter(0))
def task13():
    """
    Task 13.

    In this function you should again create and propagate a RayBundle to the output plane
    before plotting the spot plot.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the spot plot
    2. the simulation RMS

    Returns:
        tuple[Figure, float]: the spot plot and rms
    """
    bundle = RayBundle()
    sr = SphericalRefraction(
         z_0=100,
         aperture=34,
         curvature=0.03,
         n_1=1,
         n_2=1.5,
        )

    focal_point = np.round(sr.focal_point(), 1)
    op = OutputPlane(focal_point)
    elements = [sr, op]
    bundle.propagate_bundle(elements)
    plot = bundle.spot_plot()
    rms = bundle.rms()
    return plot, rms

    
@SaveOutput("task14", plot_output_indices=itemgetter(0))
def task14():
    """
    Task 14.

    In this function you will trace a number of RayBundles through the optical system and
    plot the RMS and diffraction scale dependence on input beam radii.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the diffraction scale plot
    2. the simulation RMS for input beam radius 2.5
    3. the diffraction scale for input beam radius 2.5

    Returns:
        tuple[Figure, float, float]: the plot, the simulation RMS value, the diffraction scale.
    """
    sr = SphericalRefraction()
    focal_point = np.round(sr.focal_point(), 1) 
    focal_length = focal_point - sr.z_0()
    wavelength = 588e-6
    op = OutputPlane(focal_point)
    elements = [sr, op]
    radii = np.linspace(0.1, 10, 100)
    rms_list = []
    diffraction_scale_list = []

    def scale(wavelength, focal_length, radius): 
        """Calulates the diffraction scale"""
        big_d = radius * 2
        aperture = sr.aperture()
        if aperture < radius:
            big_d = sr.aperture() * 2
        diffraction_scale = (wavelength * focal_length) / big_d
        return diffraction_scale

    for radius in radii:
        bundle = RayBundle(radius, 5, 6)
        bundle.propagate_bundle(elements)
        rms_list.append(bundle.rms())
        diffraction_scale = scale(wavelength, focal_length, radius)
        diffraction_scale_list.append(diffraction_scale)

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(radii, rms_list, label="RMS spot size")
    ax1.grid(True)
    ax1.set_xlabel("Bundle Radius (mm)")
    ax1.set_ylabel("RMS Spread of Bundle Spot (mm)")
    ax2 = ax1.twinx()
    ax2.plot(radii, diffraction_scale_list, color = "red", label="Diffraction scale")
    ax2.set_ylabel("Diffraction Scale (mm)")

    fig.legend()

    bundle = RayBundle(2.5, 5, 6)
    bundle.propagate_bundle(elements)
    return fig, bundle.rms(), scale(wavelength, focal_length, 2.5)


@SaveOutput(["task15a", "task15b"], plot_output_indices=itemgetter(0, 2))
def task15():
    """
    Task 15.

    In this function you will create plano-convex lenses in each orientation and propagate a RayBundle
    through each to their respective focal point. You should then plot the spot plot for each orientation.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the spot plot for the plano-convex system
    2. the focal point for the plano-convex lens
    3. the matplotlib figure object for the spot plot for the convex-plano system
    4  the focal point for the convex-plano lens


    Returns:
        tuple[Figure, float, Figure, float]: the spot plots and rms for plano-convex and convex-plano.
    """

    def run_lens(curvatures):
        lens = PlanoConvex(curvature = curvatures)
        focal_point = lens.focal_point()
        op = OutputPlane(focal_point)
        bundle = RayBundle()
        bundle.propagate_bundle([lens, op])
        plot = bundle.spot_plot()
        return plot, focal_point

    pc_plot, pc_focal_point = run_lens(-0.02)
    cp_plot, cp_focal_point = run_lens(0.02)

    return (pc_plot, pc_focal_point, cp_plot, cp_focal_point)



@SaveOutput("task16", plot_output_indices=itemgetter(0))
def task16():
    """
    Task 16.

    In this function you will be again plotting the radial dependence of the RMS and diffraction values
    for each orientation of your lens.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the diffraction scale plot
    2. the RMS for input beam radius 3.5 for the plano-convex system
    3. the RMS for input beam radius 3.5 for the convex-plano system
    4  the diffraction scale for input beam radius 3.5

    Returns:
        tuple[Figure, float, float, float]: the plot, RMS for plano-convex, RMS for convex-plano, diffraction scale.
    """

    cp_lens = PlanoConvex(curvature = 0.02)
    pc_lens = PlanoConvex(curvature = -0.02)
    lenses = [pc_lens, cp_lens]

    def scale(wavelength, focal_length, radius): 
        """Calulates the diffraction scale"""
        big_d = radius * 2
        aperture = lens.aperture()
        if aperture < radius:
            big_d = lens.aperture() * 2
        diffraction_scale = (wavelength * focal_length) / big_d
        return diffraction_scale

    wavelength = 588e-6
    radii = np.linspace(0.1, 10, 100)
    big_rms_list = []
    big_diffraction_scale_list = []
    rms_35mm_list = []
    diffraction_scale_35mm_list = []

    for lens in lenses:
        focal_point = lens.focal_point()
        focal_length = focal_point - lens.z_0()
        op = OutputPlane(focal_point)
        elements = [lens, op]
        rms_list = []
        diffraction_scale_list = []

        for radius in radii:
            bundle = RayBundle(radius, 5, 6)
            bundle.propagate_bundle(elements)
            rms_list.append(bundle.rms())
            diffraction_scale = scale(wavelength, focal_length, radius)
            diffraction_scale_list.append(diffraction_scale)

        big_rms_list.append(rms_list)
        big_diffraction_scale_list.append(diffraction_scale_list)
        bundle = RayBundle(3.5, 5, 6)
        bundle.propagate_bundle(elements)
        rms_35mm_list.append(bundle.rms())
        print(bundle.rms())
        diffraction_scale_35mm_list.append(scale(wavelength, focal_length, 3.5))

    fig = plt.figure()
    ax1 = fig.add_subplot()

    ax1.plot(radii, big_rms_list[1], label="P-C RMS")
    ax1.plot(radii, big_rms_list[0], label="C-P RMS")
    ax1.grid(True)
    ax1.set_xlabel("Bundle Radius (mm)")
    ax1.set_ylabel("RMS Spread of Bundle Spot (mm)")

    ax2 = ax1.twinx()
    ax2.plot(radii, big_diffraction_scale_list[1], color="red", label= "P-C diffraction scale")
    ax2.plot(radii, big_diffraction_scale_list[0], color="green", label="C-P diffraction scale",)
    ax2.set_ylabel("Diffraction Scale (mm)")
    fig.legend()

    return (fig, rms_35mm_list[0], rms_35mm_list[1], diffraction_scale_35mm_list[1])



@SaveOutput("task17", plot_output_indices=itemgetter(0))
def task17():
    """
    Task 17.

    In this function you will be first plotting the spot plot for your PlanoConvex lens with the curved
    side first (at the focal point). You will then be optimising the curvatures of a BiConvex lens
    in order to minimise the RMS spot size at the same focal point. This function should return
    the following items as a tuple in the following order:
    1. The comparison spot plot for both PlanoConvex (curved side first) and BiConvex lenses at PlanoConvex focal point.
    2. The RMS spot size for the PlanoConvex lens at focal point
    3. the RMS spot size for the BiConvex lens at PlanoConvex focal point

    Returns:
        tuple[Figure, float, float]: The combined spot plot, RMS for the PC lens, RMS for the BiConvex lens
    """

    pc_lens = PlanoConvex(curvature=0.02)
    pc_focal_point = pc_lens.focal_point()
    pc_op = OutputPlane(pc_focal_point)
    pc_bundle = RayBundle()
    pc_bundle.propagate_bundle([pc_lens, pc_op])
    pc_rms = pc_bundle.rms()

    def minimize_func(curvatures):
        """Function to be minimized"""
        curvature_1 = curvatures[0]
        curvature_2 = curvatures[1]
        bc_lens = BiConvex(curvature1 = curvature_1, curvature2 = curvature_2)
        bc_bundle = RayBundle()
        bc_bundle.propagate_bundle([bc_lens, pc_op])
        bc_rms = bc_bundle.rms()
        return bc_rms

    optimal_curvatures = minimize(minimize_func, [0.02, -0.02])
    optimized_c1 = optimal_curvatures.x[0]
    optimized_c2 = optimal_curvatures.x[1]
    bc_lens = BiConvex(curvature1 = optimized_c1, curvature2 = optimized_c2)
    bc_bundle = RayBundle()
    bc_bundle.propagate_bundle([bc_lens, pc_op])
    bc_rms = bc_bundle.rms()
    fig = pc_bundle.spot_plot()
    fig = bc_bundle.spot_plot(fig=fig)
    fig = bc_bundle.track_plot()
    return fig, pc_rms, bc_rms


@SaveOutput("task18", plot_output_indices=itemgetter(0))
def task18():
    """
    Task 18.

    In this function you will be testing your reflection modelling. Create a new SphericalReflecting surface
    and trace a RayBundle through it to the OutputPlane.This function should return
    the following items as a tuple in the following order:
    1. The track plot showing reflecting ray bundle off SphericalReflection surface.
    2. The focal point of the SphericalReflection surface.

    Returns:
        tuple[Figure, float]: The track plot, the focal point
    """
    return


@SaveOutput(["task19a", "task19b"])
def task19():
    """
    Task 19.

    In this function you will be quantifing the amount of spherical aberration.
    Create a new ConvexPlano lens and trace a RayBundle through it to the OutputPlane
    located at the focal point.
    This function should return two matplotlib figures as a tuple in the following order:
    1. The plot showing the transverse spherical aberration i.e. rms as a function of beam radius at focal point.
    2. The plot showing the longitudinal spherical aberration i.e. z intercept of a ray as a function
       of transverse distance from optical axis.

    Returns:
        tuple[Figure, Figure]: Transverse spherical aberration plot, longitudinal spherical aberration plot
    """
    return


@SaveOutput("task20")
def task20():
    """
    Task 20.

    In this function you will investigate dispersion. Create several Rays with different wavelengths.
    Plot the paths of these rays through a BiConvex lens made of BK7 glass to an OutputPlane at z=200.
    This function should return the track plot of the path of your Rays through the glass lens.

    Returns:
        Figure: The ray track plot.
    """
    return


if __name__ == "__main__":

    # Run task 8 function
    task8()
    # Run task 10 function
    FIG10 = task10()


    #Run task 11 function
    FIG11, FOCAL_POINT = task11()

    # Run task 12 function
    FIG12 = task12()

    # Run task 13 function
    FIG13, TASK13_RMS = task13()

    # Run task 14 function
    FIG14, TASK14_RMS, TASK14_DIFF_SCALE = task14()

    # Run task 15 function
    FIG15_PC, FOCAL_POINT_PC, FIG15_CP, FOCAL_POINT_CP = task15()

    # Run task 16 function
    FIG16, PC_RMS, CP_RMS, TASK16_DIFF_SCALE = task16()

    # Run task 17 function
    FIG17, CP_RMS, BICONVEX_RMS = task17()

    # Run task 18 function
    # FIG18, FOCAL_POINT = task18()

    # Run task 19 function
    # FIG19_TVSA, FIG19_LGSA = task19()

    # Run task 20 function
    # FIG20 = task20()