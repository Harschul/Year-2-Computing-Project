class BiConvex(PlanoConvex):
    """Creates a BiConvex lens"""
    def __init__(
        self,
        *,
        z_0 = 100,
        curvature1=0.02,
        curvature2=-0.02,
        n_inside=1.5168,
        n_outside=1.0,
        thickness=5.0,
        aperture=50.0,
    ):
        """Initializes the biconvex lens."""

        super().__init__(
            z_0 = z_0,
            curvature = curvature1,
            n_inside = n_inside,
            n_outside = n_outside,
            thickness = thickness,
            aperture = aperture,
        )

        self.sr_1 = elements.SphericalRefraction(
            z_0 = self.z_0(),
            aperture = self.aperture(),
            curvature = curvature1,
            n_1 = self.n_outside(),
            n_2 = self.n_inside(),
        )

        self.sr_2 = elements.SphericalRefraction(
            z_0=self.z_0() + self.thickness(),
            aperture = self.aperture(),
            curvature = curvature2,
            n_1 = self.n_inside(),
            n_2 = self.n_outside(),
        )
