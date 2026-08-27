import numpy as np

def refractive_index(lambda_um, material):
    """
    Calculates the refractive index n of a material for a given wavelength using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
        material (str): Name of the material

    Returns:
        n (float or ndarray): Refractive index
    """
    if material == "NBK7":
        return refractive_index_NBK7(lambda_um)
    elif material == "NBAF10":
        return refractive_index_NBAF10(lambda_um)
    elif material == "NSF6HT":
        return refractive_index_NSF6HT(lambda_um)
    elif material == "NSF2":
        return refractive_index_NSF2(lambda_um)
    elif material == "BaF2":
        return refractive_index_BaF2(lambda_um)
    elif material == "CaF2":
        return refractive_index_CaF2(lambda_um)
    elif material == "N_SF11":
        return refractive_index_N_SF11(lambda_um)
    elif material == "N_SF6":
        return refractive_index_N_SF6(lambda_um)
    elif material == "N_BAK4":
        return refractive_index_N_BAK4(lambda_um)
    elif material == "N_LAK22":
        return refractive_index_N_LAK22(lambda_um)
    elif material == "N_LAF21":
        return refractive_index_N_LAF21(lambda_um)
    elif material == "N_K5":
        return refractive_index_N_K5(lambda_um)
    elif material == "S_NSL3":
        return refractive_index_S_NSL3(lambda_um)
    elif material == "S_LAH79":
        return refractive_index_S_LAH79(lambda_um)
    elif material == "GFK68":
        return refractive_index_GFK68(lambda_um)
    elif material == "E_F2":
        return refractive_index_E_F2(lambda_um)
    elif material == "GFK70":
        return refractive_index_GFK70(lambda_um)
    elif material == "N_KZFS8":
        return refractive_index_N_KZFS8(lambda_um)
    elif material == "LITHO_CAF2":
        return refractive_index_LITHO_CAF2(lambda_um)
    elif material == "N_KZFS5":
        return refractive_index_N_KZFS5(lambda_um)
    elif material == "J_PSK03":
        return refractive_index_J_PSK03(lambda_um)
    elif material == "J_LASF015":
        return refractive_index_J_LASF015(lambda_um)
    elif material == "S_LAH66":
        return refractive_index_S_LAH66(lambda_um)
    elif material == "J_SF03":
        return refractive_index_J_SF03(lambda_um)
    elif material == "E_SK10":
        return refractive_index_E_SK10(lambda_um)
    elif material == "J_LAF7":
        return refractive_index_J_LAF7(lambda_um)
    elif material == "BASF6":
        return refractive_index_BASF6(lambda_um)
    elif material == "KZFH1":
        return refractive_index_KZFH1(lambda_um)
    else:
        raise ValueError("Material not recognized")

def refractive_index_NBK7(lambda_um):
    """
    Calculates the refractive index n of the material NBK-7 for a given wavelength using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
    
    Returns:
        n (float or ndarray): Refractive index
    """

    B1 = 1.03961212
    B2 = 0.231792344
    B3 = 1.01046945

    C1 = 0.00600069867
    C2 = 0.0200179144
    C3 = 103.560653

    lambda_sq = lambda_um ** 2

    n_squared = 1 + (
        (B1 * lambda_sq) / (lambda_sq - C1) +
        (B2 * lambda_sq) / (lambda_sq - C2) +
        (B3 * lambda_sq) / (lambda_sq - C3)
    )
    
    return np.sqrt(n_squared)

def refractive_index_NBAF10(lambda_um):
    """
    Calculates the refractive index n of the material NBAF10 for a given wavelength using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
    
    Returns:
        n (float or ndarray): Refractive index
    """
    B1, C1 = 1.5851495, 0.00926681282
    B2, C2 = 0.143559385, 0.0424489805
    B3, C3 = 1.08521269, 105.613573
    
    lambda_sq = lambda_um ** 2
    n_squared = 1 + (B1 * lambda_sq) / (lambda_sq - C1) \
                  + (B2 * lambda_sq) / (lambda_sq - C2) \
                  + (B3 * lambda_sq) / (lambda_sq - C3)
    return np.sqrt(n_squared)

def refractive_index_NSF6HT(lambda_um):
    """
    Calculates the refractive index n of the material NSF6HT for a given wavelength using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
    
    Returns:
        n (float or ndarray): Refractive index
    """
    B1, C1 = 1.77931763, 0.0133714182
    B2, C2 = 0.338149866, 0.0617533621
    B3, C3 = 2.08734474, 174.01759

    lambda_sq = lambda_um ** 2
    n_squared = 1 + (B1 * lambda_sq) / (lambda_sq - C1) \
                  + (B2 * lambda_sq) / (lambda_sq - C2) \
                  + (B3 * lambda_sq) / (lambda_sq - C3)
    return np.sqrt(n_squared)

def refractive_index_NSF2(lambda_um):
    """
    Calculates the refractive index n of the material NSF2 for a given wavelength using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
    
    Returns:
        n (float or ndarray): Refractive index
    """
    B1, C1 = 1.47343127, 0.0109019098
    B2, C2 = 0.163681849, 0.0585683687
    B3, C3 = 1.36920899, 127.404933

    lambda_sq = lambda_um ** 2
    n_squared = 1 + (B1 * lambda_sq) / (lambda_sq - C1) \
                  + (B2 * lambda_sq) / (lambda_sq - C2) \
                  + (B3 * lambda_sq) / (lambda_sq - C3)
    return np.sqrt(n_squared)

def refractive_index_BaF2(lambda_um):
    """
    Calculates the refractive index n of BaF2 using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of BaF2.
    """
    A = 0.33973
    B1, C1 = 0.81070, 0.10065**2
    B2, C2 = 0.19652, 29.87**2
    B3, C3 = 4.52469, 53.82**2

    lambda_sq = lambda_um**2

    n_squared = 1 + A + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)

def refractive_index_CaF2(lambda_um):
    """
    Calculates the refractive index n of CaF2 using the Sellmeier equation.

    Valid for wavelengths from 0.15 to 12.0 µm.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of CaF2.
    """
    A = 0.33973
    B1, C1 = 0.69913, 0.09374**2
    B2, C2 = 0.11994, 21.18**2
    B3, C3 = 4.35181, 38.46**2

    lambda_sq = lambda_um**2

    n_squared = 1 + A + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)

def refractive_index_Ge(lambda_um):
    """
    Calculates the refractive index n of Germanium (Ge) using the Sellmeier equation.

    Valid for wavelengths from 2.5 to 12.0 µm.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of Ge.
    """
    A = 8.28156
    B1, C1 = 6.72888, 0.44105
    B2, C2 = 0.21307, 3870.1

    lambda_sq = lambda_um**2

    n_squared = 1 + A + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2)
    )

    return np.sqrt(n_squared)

def refractive_index_MgF2(lambda_um):
    """
    Calculates the refractive index n of MgF2 using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of MgF2.
    """
    B1, C1 = 0.48755108, 0.04338408**2
    B2, C2 = 0.39875031, 0.09461442**2
    B3, C3 = 2.3120353, 23.793604**2

    lambda_sq = lambda_um**2

    n_squared = 1 + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)

def refractive_index_N_SF11(lambda_um):
    """
    Calculates the refractive index n of N-SF11 using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of N-SF11.
    """
    lambda_sq = lambda_um**2

    # Sellmeier coefficients (from provided equation)
    B1, C1 = 1.73759695, 0.013188707
    B2, C2 = 0.313747346, 0.0623068142
    B3, C3 = 1.89878101, 155.23629

    n_squared = 1 + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)

def refractive_index_N_SF6(lambda_um):
    """
    Calculates the refractive index n of N-SF6 using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of N-SF6.
    """
    lambda_sq = lambda_um**2

    B1, C1 = 1.779317630, 0.013371418
    B2, C2 = 0.338149866, 0.0617533621
    B3, C3 = 2.087344740, 174.01759000

    n_squared = 1 + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)

def refractive_index_N_BAK4(lambda_um):
    """
    Calculates the refractive index n of N-BAK4 using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of N-BAK4.
    """
    lambda_sq = lambda_um**2

    B1, C1 = 1.288346420, 0.007799806
    B2, C2 = 0.132817724, 0.0315631177
    B3, C3 = 0.945395373, 105.96587500

    n_squared = 1 + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)

def refractive_index_N_LAK22(lambda_um):
    """
    Calculates the refractive index n of N-LAK22 using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of N-LAK22.
    """
    lambda_sq = lambda_um**2

    B1, C1 = 1.142297810, 0.005857786
    B2, C2 = 0.535138441, 0.0198546147
    B3, C3 = 1.040883850, 100.83401700

    n_squared = 1 + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)

def refractive_index_N_LAF21(lambda_um):
    """
    Calculates the refractive index n of N-LAF21 using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of N-LAF21.
    """
    lambda_sq = lambda_um**2

    B1, C1 = 1.871345290, 0.009333223
    B2, C2 = 0.250783010, 0.0345637762
    B3, C3 = 1.220486390, 83.24048660

    n_squared = 1 + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)

def refractive_index_N_K5(lambda_um):
    """
    Calculates the refractive index n of N-K5 using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm).

    Returns:
        n (float or ndarray): Refractive index of N-K5.
    """
    lambda_sq = lambda_um**2

    B1, C1 = 1.085118330, 0.006610995
    B2, C2 = 0.199562005, 0.0241108660
    B3, C3 = 0.930511663, 111.98277700

    n_squared = 1 + (
        B1 * lambda_sq / (lambda_sq - C1) +
        B2 * lambda_sq / (lambda_sq - C2) +
        B3 * lambda_sq / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)

def refractive_index_S_NSL3(lambda_um):
    """
    Calculates the refractive index n of the material S-NSL3 for a given wavelength using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
    
    Returns:
        n (float or ndarray): Refractive index
    """

    B1 = 8.82514764E-01
    B2 = 3.89271907E-01
    B3 = 1.10693448E+00

    C1 = 4.64504582E-03
    C2 = 2.00551397E-02
    C3 = 1.36234339E+02

    lambda_sq = lambda_um ** 2

    n_squared = 1 + (
        (B1 * lambda_sq) / (lambda_sq - C1) +
        (B2 * lambda_sq) / (lambda_sq - C2) +
        (B3 * lambda_sq) / (lambda_sq - C3)
    )
    
    return np.sqrt(n_squared)



def refractive_index_S_LAH79(lambda_um):
    """
    Calculates the refractive index n of S-LAH79 for a given wavelength using
    the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)

    Returns:
        n (float or ndarray): Refractive index
    """

    B1 = 2.32557148E+00
    B2 = 0.507967133
    B3 = 2.43087198

    C1 = 0.0132895208
    C2 = 0.0528335449
    C3 = 161.122408

    lambda_sq = lambda_um ** 2

    n_squared = 1 + (
        (B1 * lambda_sq) / (lambda_sq - C1) +
        (B2 * lambda_sq) / (lambda_sq - C2) +
        (B3 * lambda_sq) / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)


def refractive_index_GFK68(lambda_um):
    """
    Calculates the refractive index n of GFK68 for a given wavelength using
    the dispersion equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)

    Returns:
        n (float or ndarray): Refractive index
    """

    A0 = 2.4994867
    A1 = -0.0059285872
    A2 = 0.012743526
    A3 = 0.00015469636
    A4 = 3.9947612e-06
    A5 = -1.7603560e-07

    lambda_sq = lambda_um ** 2

    n_squared = (
        A0 +
        A1 * lambda_sq +
        A2 * lambda_um ** -2 +
        A3 * lambda_um ** -4 +
        A4 * lambda_um ** -6 +
        A5 * lambda_um ** -8
    )

    return np.sqrt(n_squared)


def refractive_index_E_F2(lambda_um):
    """
    Calculates the refractive index n of E-F2 for a given wavelength using
    the dispersion equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)

    Returns:
        n (float or ndarray): Refractive index
    """

    A0 = 2.55739282
    A1 = -0.0107951229
    A2 = 0.0216807328
    A3 = 0.00105165979
    A4 = -5.36309816e-05
    A5 = 7.7466326e-06

    lambda_sq = lambda_um ** 2

    n_squared = (
        A0 +
        A1 * lambda_sq +
        A2 * lambda_um ** -2 +
        A3 * lambda_um ** -4 +
        A4 * lambda_um ** -6 +
        A5 * lambda_um ** -8
    )

    return np.sqrt(n_squared)


def refractive_index_GFK70(lambda_um):
    """
    Calculates the refractive index n of GFK70 for a given wavelength using
    the dispersion equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)

    Returns:
        n (float or ndarray): Refractive index
    """

    A0 = 2.4293935
    A1 = -0.0057591097
    A2 = 0.011512013
    A3 = 0.0001324924
    A4 = 7.3250033e-06
    A5 = -5.7195445e-07

    lambda_sq = lambda_um ** 2

    n_squared = (
        A0 +
        A1 * lambda_sq +
        A2 * lambda_um ** -2 +
        A3 * lambda_um ** -4 +
        A4 * lambda_um ** -6 +
        A5 * lambda_um ** -8
    )

    return np.sqrt(n_squared)


def refractive_index_N_KZFS8(lambda_um):
    """
    Calculates the refractive index n of N-KZFS8 for a given wavelength using
    the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)

    Returns:
        n (float or ndarray): Refractive index
    """

    B1 = 1.62693651
    B2 = 0.24369876
    B3 = 1.62007141

    C1 = 0.010880863
    C2 = 0.0494207753
    C3 = 131.009163

    lambda_sq = lambda_um ** 2

    n_squared = 1 + (
        (B1 * lambda_sq) / (lambda_sq - C1) +
        (B2 * lambda_sq) / (lambda_sq - C2) +
        (B3 * lambda_sq) / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)


def refractive_index_LITHO_CAF2(lambda_um):
    """
    Calculates the refractive index n of LITHO-CAF2 for a given wavelength
    using the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)

    Returns:
        n (float or ndarray): Refractive index
    """

    B1 = 0.617617011
    B2 = 0.421117656
    B3 = 3.79711183

    C1 = 0.00275381936
    C2 = 0.0105900875
    C3 = 1182.67444

    lambda_sq = lambda_um ** 2

    n_squared = 1 + (
        (B1 * lambda_sq) / (lambda_sq - C1) +
        (B2 * lambda_sq) / (lambda_sq - C2) +
        (B3 * lambda_sq) / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)


def refractive_index_N_KZFS5(lambda_um):
    """
    Calculates the refractive index n of N-KZFS5 for a given wavelength using
    the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)

    Returns:
        n (float or ndarray): Refractive index
    """

    B1 = 1.47460789
    B2 = 0.193584488
    B3 = 1.265899740

    C1 = 0.00986143816
    C2 = 0.0445477583
    C3 = 106.436258

    lambda_sq = lambda_um ** 2

    n_squared = 1 + (
        (B1 * lambda_sq) / (lambda_sq - C1) +
        (B2 * lambda_sq) / (lambda_sq - C2) +
        (B3 * lambda_sq) / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)


def refractive_index_J_PSK03(lambda_um):
    """
    Calculates the refractive index n of J-PSK03 for a given wavelength using
    the dispersion equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)

    Returns:
        n (float or ndarray): Refractive index
    """

    A0 = 2.53267453
    A1 = -0.00950416844
    A2 = -0.000106883723
    A3 = 0.013439736
    A4 = 0.000141770605
    A5 = 4.7304388e-06
    A6 = -8.6200083e-08

    lambda_sq = lambda_um ** 2

    n_squared = (
        A0 +
        A1 * lambda_sq +
        A2 * lambda_um ** 4 +
        A3 * lambda_um ** -2 +
        A4 * lambda_um ** -4 +
        A5 * lambda_um ** -6 +
        A6 * lambda_um ** -8
    )

    return np.sqrt(n_squared)


def refractive_index_J_LASF015(lambda_um):
    """
    Calculates the refractive index n of J-LASF015 for a given wavelength
    using the dispersion equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)

    Returns:
        n (float or ndarray): Refractive index
    """

    A0 = 3.17452404
    A1 = -0.0132156517
    A2 = -0.000165919934
    A3 = 0.0276472367
    A4 = 0.000483338934
    A5 = 1.20380702e-05
    A6 = 6.02649728e-07

    lambda_sq = lambda_um ** 2

    n_squared = (
        A0 +
        A1 * lambda_sq +
        A2 * lambda_um ** 4 +
        A3 * lambda_um ** -2 +
        A4 * lambda_um ** -4 +
        A5 * lambda_um ** -6 +
        A6 * lambda_um ** -8
    )

    return np.sqrt(n_squared)


def refractive_index_S_LAH66(lambda_um):
    """
    Calculates the refractive index n of S-LAH66 for a given wavelength using
    the Sellmeier equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)

    Returns:
        n (float or ndarray): Refractive index
    """

    B1 = 1.39280586
    B2 = 0.679577094
    B3 = 1.38702069

    C1 = 0.00608475118
    C2 = 0.0233925351
    C3 = 95.8354094

    lambda_sq = lambda_um ** 2

    n_squared = 1 + (
        (B1 * lambda_sq) / (lambda_sq - C1) +
        (B2 * lambda_sq) / (lambda_sq - C2) +
        (B3 * lambda_sq) / (lambda_sq - C3)
    )

    return np.sqrt(n_squared)


def refractive_index_J_SF03(lambda_um):
    """
    Calculates the refractive index n of J-SF03 for a given wavelength using
    the dispersion equation.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)

    Returns:
        n (float or ndarray): Refractive index
    """

    A0 = 3.25089291
    A1 = -0.013324411
    A2 = 0.0484040988
    A3 = 0.0032638368
    A4 = -0.000401470701
    A5 = 0.000116583198
    A6 = -1.27242455e-05
    A7 = 6.96171808e-07

    lambda_sq = lambda_um ** 2

    n_squared = (
        A0 +
        A1 * lambda_sq +
        A2 * lambda_um ** -2 +
        A3 * lambda_um ** -4 +
        A4 * lambda_um ** -6 +
        A5 * lambda_um ** -8 +
        A6 * lambda_um ** -10 +
        A7 * lambda_um ** -12
    )

    return np.sqrt(n_squared)

def refractive_index_E_SK10(lambda_um):
    """
    Calculates the refractive index n of the material E-SK10
    for a given wavelength.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
    
    Returns:
        n (float or ndarray): Refractive index
    """

    A0 = 2.58912326
    A1 = -0.0100115186
    A2 = 0.0156165805
    A3 = 0.000328758605
    A4 = -9.84858579e-06
    A5 = 7.82877169e-07

    lambda_sq = lambda_um ** 2

    n_squared = (
        A0 +
        A1 * lambda_sq +
        A2 / lambda_sq +
        A3 / lambda_sq**2 +
        A4 / lambda_sq**3 +
        A5 / lambda_sq**4
    )
    
    return np.sqrt(n_squared)



def refractive_index_J_LAF7(lambda_um):
    """
    Calculates the refractive index n of the material J-LAF7
    for a given wavelength.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
    
    Returns:
        n (float or ndarray): Refractive index
    """

    A0 = 2.96739544
    A1 = -0.0118139418
    A2 = -0.000133628078
    A3 = 0.0310749099
    A4 = 0.000654571893
    A5 = 9.85567905e-05
    A6 = -8.8311254e-06
    A7 = 8.38843732e-07

    lambda_sq = lambda_um ** 2

    n_squared = (
        A0 +
        A1 * lambda_sq +
        A2 * lambda_sq**2 +
        A3 / lambda_sq +
        A4 / lambda_sq**2 +
        A5 / lambda_sq**3 +
        A6 / lambda_sq**4 +
        A7 / lambda_sq**5
    )
    
    return np.sqrt(n_squared)


def refractive_index_BASF6(lambda_um):
    """
    Calculates the refractive index n of the material BASF6
    for a given wavelength.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
    
    Returns:
        n (float or ndarray): Refractive index
    """

    A0 = 2.718163
    A1 = -0.01334622
    A2 = 0.02057831
    A3 = 0.0009283221
    A4 = -1.411159e-05
    A5 = 1.659476e-06

    lambda_sq = lambda_um ** 2

    n_squared = (
        A0 +
        A1 * lambda_sq +
        A2 / lambda_sq +
        A3 / lambda_sq**2 +
        A4 / lambda_sq**3 +
        A5 / lambda_sq**4
    )
    
    return np.sqrt(n_squared)


def refractive_index_KZFH1(lambda_um):
    """
    Calculates the refractive index n of the material KZFH1
    for a given wavelength.

    Args:
        lambda_um (float or ndarray): Wavelength in micrometers (µm)
    
    Returns:
        n (float or ndarray): Refractive index
    """

    A0 = 2.547175
    A1 = -0.01287115
    A2 = 0.01838384
    A3 = 0.0005862006
    A4 = -1.624561e-05
    A5 = 2.266107e-06

    lambda_sq = lambda_um ** 2

    n_squared = (
        A0 +
        A1 * lambda_sq +
        A2 / lambda_sq +
        A3 / lambda_sq**2 +
        A4 / lambda_sq**3 +
        A5 / lambda_sq**4
    )
    
    return np.sqrt(n_squared)