import numpy as np
from typing import Union, Tuple


def generate_tiles(
    samples: int, lines: int, xtile=Union[None, int], ytile=Union[None, int]
) -> Tuple:
    """
    Generates a list of tile indices for a 2D array. Taken from
    https://github.com/GeoscienceAustralia/wagl/blob/water-atcor/wagl/tiling.py
    Parameters
    ----------
    samples : int
        An integer expressing the total number of samples (columns)
        in an array.
    lines : int
        An integer expressing the total number of lines (rows)
        in an array.
    xtile : int or None
        (Optional) The desired size of the tile in the x-direction.
        Default is all samples
    ytile : int or None
        (Optional) The desired size of the tile in the y-direction.
        Default is min(100, lines) lines.
    Returns
    -------
        Each tuple in the generator contains
        ((ystart,yend),(xstart,xend)).
    Examples
    --------
        >>> from wagl.tiling import generate_tiles
        >>> tiles = generate_tiles(8624, 7567, xtile=1000, ytile=400)
        >>> for tile in tiles:
        >>>     # A rasterio dataset
        >>>     subset = rio_ds.read([4, 3, 2], window=tile)
        >>>     # Or simply move the tile window across an array
        >>>     subset = array[tile]  # 2D
        >>>     subset = array[:,tile[0],tile[1]]  # 3D
    """

    def create_tiles(samples, lines, xstart, ystart):
        """
        Creates a generator object for the tiles.
        """
        for ystep in ystart:
            if ystep + ytile < lines:
                yend = ystep + ytile
            else:
                yend = lines
            for xstep in xstart:
                if xstep + xtile < samples:
                    xend = xstep + xtile
                else:
                    xend = samples
                yield (slice(ystep, yend), slice(xstep, xend)) # yield is like return, except it returns a generator (iterators) that u can only iterate over once. they do not store all the valuesin memory, 
                #they generate values on the fly. it's useful if a function returns a huge set of values that u will only need to read once

    # check for default or out of bounds
    if xtile is None or xtile < 0:
        xtile = samples
    if ytile is None or ytile < 0:
        ytile = min(100, lines)

    xstart = np.arange(0, samples, xtile)
    ystart = np.arange(0, lines, ytile)

    tiles = create_tiles(samples, lines, xstart, ystart)

    return tiles

def calc_pfresnel(
    w: Union[float, np.ndarray], n_sw: float = 1.34
) -> Union[np.ndarray, float]:
    """
    Calculate the fresnel reflection of sunglint at the water's surface
    Parameters
    ----------
    w : float or numpy.ndarray (np.float32/64)
        Angle of incidence of a light ray at the water surface (radians)
    n_sw : float
        Refractive index of sea-water (wavelength independent)
    Returns
    -------
    p_fresnel : numpy.ndarray (np.float32/64)
        The fresnel reflectance
    ---------------------
    For air-incident case, when light hits the air-water surface, the light gets reflected towards the sky (reflected), the rest is transmitted into the water
    The converse is true for water-incident case. This phenomenon is described by the snell's law
    For the air-incident case, snell's law is sin(theta_reflected) = n_sw*sin(theta_transmitted)
    theta_reflected = angle of incidence = w
    The fresnel reflectance lies in the interval [0,1] and gives the fraction of photos incident in a narrow beam that is reflected by the surface
    Reflectance is 0.02 to 0.03 for rays with incident angles of less than 30deg
    For air-incident rays, the reflectance does not exceed 0.1 until the angle of incidence is greater than 65
    #####################
    Credits: https://github.com/GeoscienceAustralia/sun-glint-correction/blob/develop/sungc/algorithms.py
    #####################
    """
    w_pr = np.asin(np.sin(w)/n_sw) #transmitted reflection based on snells law
    p_fres = 0.5*((np.sin(w-w_pr)/np.sin(w+w_pr))**2 + (np.tan(w-w_pr)/np.tan(w+w_pr))**2) #based on fresnel's formula: the reflectance of the air-water surface, which holds if w !=0

    return p_fres

def coxmunk_backend(
    view_zenith: np.ndarray,
    solar_zenith: np.ndarray,
    relative_azimuth: np.ndarray,
    wind_speed: float,
    return_fresnel: bool = False,
) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Estimates the wavelength-independent sunglint reflectance using the
    Cox and Munk (1954) algorithm. Here the wind direction is not taken
    into account.
    Parameters
    ----------
    view_zenith : np.ndarray
        sensor view-zenith angle image
        (units of image data: degrees)
    solar_zenith : np.ndarray
        solar zenith angle image
        (units of image data: degrees)
    relative_azimuth : np.ndarray
        relative azimuth angle between sensor and sun image
        image (units of image data: degrees)
    wind_speed : float
        Wind speed (m/s)
    return_fresnel : bool
        Return fresnel reflectance array
    Returns
    -------
    p_glint : numpy.ndarray (np.float32/64)
        Estimated sunglint reflectance
    p_fresnel : None or numpy.ndarray (np.float32/64)
        Fresnel reflectance of sunglint. Useful for debugging
        if return_fresnel=False then p_fresnel=None
        if return_fresnel=True  then p_fresnel=numpy.ndarray
    Raises
    ------
    ValueError:
        * if input arrays are not two-dimensional
        * if dimension mismatch
        * if wind_speed < 0
    #####################
    Credits: https://github.com/GeoscienceAustralia/sun-glint-correction/blob/develop/sungc/algorithms.py
    #####################
    """
    if (
        (view_zenith.ndim != 2) # camera position
        or (solar_zenith.ndim != 2) 
        or (relative_azimuth.ndim != 2)
    ):
        raise ValueError("\ninput arrays must be two dimensional")

    nrows, ncols = relative_azimuth.shape

    if (
        (nrows != solar_zenith.shape[0])
        or (nrows != view_zenith.shape[0])
        or (ncols != solar_zenith.shape[1])
        or (ncols != view_zenith.shape[1])
    ):
        raise ValueError("\nDimension mismatch")

    if wind_speed < 0:
        raise ValueError("\nwind_speed must be greater than 0 m/s")

    # create output array
    p_glint = np.zeros([nrows, ncols], order="C", dtype=view_zenith.dtype)

    p_fresnel = None
    if return_fresnel: #if return_fresnel is True
        p_fresnel = np.zeros([nrows, ncols], order="C", dtype=view_zenith.dtype)

    # Define parameters needed for the wind-direction-independent model
    pi_ = np.pi  # noqa # pylint: disable=unused-variable
    n_sw = 1.34  # refractive index of seawater
    deg2rad = np.pi / 180.0
    sigma2 = 0.003 + 0.00512 * wind_speed  # noqa # pylint: disable=unused-variable

    # This implementation creates 16 float32/64 numpy.ndarray's with
    # the same dimensions as the inputs (view_zenith, solar_zenith,
    # relative_azimuth). If the dimensions of these inputs are very
    # large, then a memory issue may arise. A better way would be to
    # iterate through tiles/blocks of these input arrays. This will
    # cause a slightly longer processing time
    tiles = generate_tiles(samples=ncols, lines=nrows, xtile=256, ytile=256)
    for t_ix in tiles:

        phi_raz = np.copy(relative_azimuth[t_ix])
        phi_raz[phi_raz > 180.0] -= 360.0
        phi_raz *= deg2rad

        theta_szn = solar_zenith[t_ix] * deg2rad  # solar zenith in deg
        theta_vzn = view_zenith[t_ix] * deg2rad  # view zenith in deg

        cos_theta_szn = np.cos(theta_szn)  # noqa # pylint: disable=unused-variable
           
        cos_theta_vzn = np.cos(theta_vzn) # noqa # pylint: disable=unused-variable

        # compute cos(w)
        # w = angle of incidence of a light ray at the water surface
        # use numexpr instead
        cos_2w = cos_theta_szn*cos_theta_vzn + np.sin(theta_szn)*np.sin(theta_vzn)*np.sin(phi_raz)

        # use trig. identity, cos(x/2) = +/- sqrt{ [1 + cos(x)] / 2 }
        # hence,
        # cos(2w/2) = cos(w) = +/- sqrt{ [1 + cos(2w)] / 2 }
        cos_w = ((1.0 + cos_2w) / 2.0) ** 0.5  # noqa # pylint: disable=unused-variable

        # compute cos(B), where B = beta;  numpy.ndarray
        cos_b = (cos_theta_szn + cos_theta_vzn) / (2.0 * cos_w)  # noqa # pylint: disable=unused-variable

        # compute tan(B)^2 = sec(B)^2 -1;  numpy.ndarray
        tan_b2 = (1.0 / (cos_b ** 2.0)) - 1.0  # noqa # pylint: disable=unused-variable

        # compute surface slope distribution:
        dist_SurfSlope = 1.0 / (pi_ * sigma2) * np.exp(-1.0 * tan_b2 / sigma2)

        # calculcate the Fresnel reflectance, numpy.ndarray
        p_fr = calc_pfresnel(w=np.accos(cos_w), n_sw=n_sw)
        if return_fresnel:
            p_fresnel[t_ix] = p_fr

        # according to Kay et al. (2009):
        # "n_sw varies with wavelength from 1.34 to 1.35 for sea-water
        #  givind p_fresnel from 0.021 to 0.023 at 20 degree incidence
        #  and 0.060 to 0.064 at 60 degree. The variation with angle is
        #  much greater than that with wavelength"

        # calculate the glint reflectance image, numpy.ndarray
        p_glint[t_ix] = pi_ * p_fr * dist_SurfSlope/ (4.0 * cos_theta_szn * cos_theta_vzn * (cos_b ** 4))

    return p_glint, p_fresnel

def cox_munk(
        self,
        vis_bands: List[str],
        vzen_band: str,
        szen_band: str,
        razi_band: str,
        wind_speed: float = 5,
    ) -> xr.Dataset:
        """
        Performs the wind-direction-independent Cox and Munk (1954)
        sunglint correction on the specified visible bands. This
        algorithm is suitable for spatial resolutions between
        100 - 1000 metres (Kay et al., 2009). Fresnel reflectance
        of sunglint is assumed to be wavelength-independent.
        Cox, C., Munk, W. 1954. Statistics of the Sea Surface Derived
        from Sun Glitter. J. Mar. Res., 13, 198-227.
        Cox, C., Munk, W. 1954. Measurement of the Roughness of the Sea
        Surface from Photographs of the Suns Glitter. J. Opt. Soc. Am.,
        44, 838-850.
        Kay, S., Hedley, J. D., Lavender, S. 2009. Sun Glint Correction
        of High and Low Spatial Resolution Images of Aquatic Scenes:
        a Review of Methods for Visible and Near-Infrared Wavelengths.
        Remote Sensing, 1, 697-730; doi:10.3390/rs1040697
        Parameters
        ----------
        vis_bands : list
            A list of band numbers in the visible that will be deglinted
        vzen_band : str
            The variable name in the xarray.Datatset for the satellite
            view-zenith band
        szen_band : str
            The variable name in the xarray.Dataset for the solar-zenith
            band
        razi_band : str
            The variable name in the xarray.Dataset for the relative
            azimuth band
        wind_speed : float
            wind speed (m/s)
        Returns
        -------
        deglinted_dxr : xr.Dataset
            xarray.Dataset object containing the deglinted vis_bands
        Raises
        ------
        ValueError:
            * If any of the input variable names do not exist
            * if input arrays are not two-dimensional
            * if dimension mismatch
            * if wind_speed < 0
        """