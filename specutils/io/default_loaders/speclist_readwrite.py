# -*- coding: utf-8 -*-
#
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#  Created on 20-Sep-2022
#
#  @author: tbowers

""" Specutils writer for SpectrumList objects

Specutils does not contain a built-in writer for the SpectrumList class (and
hence no reader for a thing that doesn't exist), so this custom writer (and its
associated reader) does it for us.

When in doubt, build it yourself.  I'm sure this is pretty kludgy, but it seems
to work for my needs.
"""

import astropy.io.fits
import astropy.table
import astropy.units
from specutils import SpectrumList, Spectrum1D
from specutils.io.parsing_utils import read_fileobj_or_hdulist
from specutils.io.registers import custom_writer, data_loader
from specutils.version import version

# =============================================================================#
# Writing Function
@custom_writer("speclist-writer", dtype=SpectrumList)
def speclist_fits(speclist, file_name, overwrite=False, **kwargs):
    """FITS Writer for specutils SpectrumList class

    Custom writer for the specutils SpectrumList class that creates a
    multiextension FITS file where each extension corresponds to one of the
    constituent Spectrum1D files in the SpectrumList.

    To use this writer, the user must specify ``format="speclist-writer",``
    in the ``spectrumlist_obj.write()`` command.

    Parameters
    ----------
    speclist : ``specutils.SpectrumList``
        The SpectrumList instance to write out
    file_name : ``str``
        Filename to write the instance to
    overwrite : ``bool``, optional
        Overwrite an existing file?  (Default: False)
    """
    # Create a primary HDU with some basic information
    pri_hdu = astropy.io.fits.PrimaryHDU()
    pri_hdu.header["SPECUTIL"] = "This file was written by specutils"
    pri_hdu.header["VERSSPC"] = (version, "Specutil version")
    pri_hdu.header["SUWRITER"] = ("speclist-writer", "Specutil Custom Writer")
    pri_hdu.header["SUCLASS"] = ("SpectrumList", "Specutil Input Class Type")
    pri_hdu.header["N_SPEC"] = (len(speclist), "Number of Spectrum1D Objects Contained")
    pri_hdu.header["SUFNAME"] = (file_name, "Name of this Specutil File")

    # Create the HDUList
    hdul = astropy.io.fits.HDUList(pri_hdu)

    # Load each subsequent Spectrum1D into its own bintable HDU
    for spectrum in speclist:
        # Shortcut to the FITS header information
        hdr = spectrum.meta["header"]
        # Create the table
        tab = astropy.table.Table(
            [spectrum.spectral_axis, spectrum.flux], names=("spectral_axis", "flux")
        )
        # And stuff it into a BinTableHDU
        spec_hdu = astropy.io.fits.BinTableHDU(
            tab, hdr, name=hdr["FILENAME"].replace(".fits", "")
        )
        # Append
        hdul.append(spec_hdu)

    # Write out the hole HDUList to FITS
    hdul.writeto(file_name, overwrite=overwrite)


# =============================================================================#
# Reading Functions
def identify_specutil_speclist(origin, *args, **kwargs):
    """
    Check whether the given file is a specutils SpectrumList FITS file written
    by the custom writer ``speclist_fits()`` in this module.
    """
    with read_fileobj_or_hdulist(*args, **kwargs) as hdulist:
        return (
            "SPECUTIL" in hdulist[0].header
            and len(hdulist) > 1
            and (
                isinstance(hdulist[1], astropy.io.fits.BinTableHDU)
                and hdulist[0].header.get("SUCLASS") == "SpectrumList"
            )
        )


@data_loader(
    "Specutils SpectrumList Output",
    identifier=identify_specutil_speclist,
    extensions=["fits"],
    priority=10,
    dtype=SpectrumList,
)
def specutils_speclist_loader(filename, **kwargs):
    """
    Loader for Specutils SlectrumList multiple-spectra files.

    Parameters
    ----------
    filename : ``str``
        The path to the FITS file

    Returns
    -------
    ``specutils.SpectrumList``
        The spectrum contained in the file.
    """
    with astropy.io.fits.open(filename, **kwargs) as hdulist:

        # Loop through the HDUs looking for spectra
        spectra = []
        for hdu in hdulist:

            # Skip non-spectral HDUs
            # All SpectrumList spectra HDUs have EXTNAME starting with '20YYMMDD'
            if "EXTNAME" not in hdu.header or hdu.header["EXTNAME"][:2] != "20":
                continue

            # Read in this BinTable as a Quantity-based table
            spec_obj = astropy.table.QTable.read(hdu)

            # Set meta as the header for this BinTable
            meta = {"header": hdu.header}

            # Package the spectrum as a Spectrum1d() object
            spec = Spectrum1D(
                flux=spec_obj["flux"],
                uncertainty=spec_obj["uncertainty"]
                if "uncertainty" in spec_obj.colnames
                else None,
                meta=meta,
                spectral_axis=spec_obj["spectral_axis"],
                velocity_convention="doppler_optical",
                bin_specification="centers",
            )
            spectra.append(spec)

        # Package and return
        return SpectrumList(spectra)
