"""
a3_power_thermal_tool.py
========================

Space Systems Engineering -- Assignment 03 (Power & Thermal) design aid.

What this tool does
-------------------
It walks you through the power and thermal sizing of your lunar rover in the
same order as the lectures (Lecture 06: power; Lecture 08 Part 1: thermal), and
then checks the places where the two subsystems collide:

    Step 1  ConOps window      -> the lowest and highest Sun elevation you see
    Step 2  Power budget       -> average load with margin (Lecture 06, WE "budget"), WE: Worked Example
    Step 3  Solar array        -> area from the effective efficiency (Lecture 06, WE1)
    Step 4  Battery            -> capacity from the worst peak deficit (Lecture 06, WE2)
    Step 5  Radiator, hot case -> area at the noon hot case (Lecture 08, WE1-WE2)
    Step 6  Heater, cold case  -> heater power at dawn (Lecture 08, WE3)
    Step 7  Integration checks -> roof area, dawn energy balance, heater fed back
                                  into the power budget, mass fed back to A01

Every function prints (or returns) its intermediate steps so you can compare
them with your hand calculation. The tool is there to let you explore the
trade space -- you are assessed on understanding WHY the numbers come out as
they do, not on the tool output itself.

UNITS -- read this before you change anything
---------------------------------------------
Every variable name carries its unit as a suffix:
    _W     watts                 _Wh   watt-hours
    _m2    square metres         _kg   kilograms
    _K     kelvin                _degC degrees Celsius
    _deg   degrees (angles)      _days Earth-days
    _V     volts                 _WK   watts per kelvin (conductance)

Temperatures that go inside sigma*T^4 are ALWAYS in kelvin. Use the helper
c_to_k() to convert. Putting degrees Celsius inside T^4 is the single most
common A3 error -- the tool will not stop you from doing it if you bypass
the helper, so don't.

Assumptions (state them in your report if you rely on them)
-----------------------------------------------------------
* The rover sits at the lunar equator, and the Sun moves in the east-west
  plane (sub-solar latitude ~0 deg; the Moon's 1.5 deg obliquity is ignored).
* The Sun moves at 360 deg per synodic month = ~12.19 deg per Earth-day,
  i.e. ~0.51 deg per hour (Lecture 08: "0.5 deg per hour").
* The environment changes slowly, so each case is a steady state
  (Lecture 08, slide 20).
* Up-facing surfaces see no ground (view factor F_ground = 0); vertical
  surfaces see half ground (F_ground = 0.5) (Lecture 08, slide 14).
* All representative values are teaching defaults. Replace them with your
  own datasheet values and cite them.

Run it:
    python a3_power_thermal_tool.py
which runs the lecture reference design. Then copy the `RoverDesign(...)`
block at the bottom, put in YOUR numbers, and run again.

Requires: Python 3.9+ and nothing else (no numpy needed).

OBS1: SLIDE NUMBERS THAT i REFERE HERE MIGHT NOT BE THE CORRECT ONES AS I WROTE THIS CODE BEFORE FINALIZING THE SLIDES,
BUT YOU SHOULD BE ABLE FIND THE NECESSARY INFO FROM LECTURES 06 AND 08 YOURSELF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Physical constants (units in the name, as everywhere else)
# ---------------------------------------------------------------------------
SIGMA_W_m2K4 = 5.67e-8          # Stefan-Boltzmann constant [W/(m^2 K^4)]
SOLAR_FLUX_W_m2 = 1361.0        # Solar constant at 1 AU, same at the Moon [W/m^2]
LUNAR_ALBEDO = 0.12             # Bond albedo of the regolith [-] (Lecture 08)
SYNODIC_MONTH_days = 29.53      # One lunar day-night cycle [Earth-days]
SUN_RATE_deg_per_day = 360.0 / SYNODIC_MONTH_days   # ~12.19 deg per Earth-day


def c_to_k(temp_degC: float) -> float:
    """Convert a temperature from degrees Celsius to kelvin.

    Args:
        temp_degC: temperature [deg C]
    Returns:
        temperature [K]
    """
    return temp_degC + 273.15


def k_to_c(temp_K: float) -> float:
    """Convert a temperature from kelvin to degrees Celsius."""
    return temp_K - 273.15


# ===========================================================================
# STEP 1 -- the ConOps window and the Sun elevation it gives you
# ===========================================================================

def sun_elevation_deg(time_from_local_noon_days: float) -> float:
    """Sun elevation above the horizon at the lunar equator.

    At local noon the Sun is at the zenith (90 deg). It drops by
    ~12.19 deg every Earth-day either side of noon.

    Args:
        time_from_local_noon_days: Earth-days relative to local noon
            (negative = morning, positive = afternoon).
    Returns:
        Sun elevation e [deg]; 0 or below means the Sun has set.
    """
    return 88.5 - SUN_RATE_deg_per_day * abs(time_from_local_noon_days)


def time_from_noon_for_elevation_days(elevation_deg: float) -> float:
    """Inverse of sun_elevation_deg: how far from noon a given elevation is.

    Returns the magnitude in Earth-days (apply a minus sign for the morning).
    """
    return (88.5 - elevation_deg) / SUN_RATE_deg_per_day


# ===========================================================================
# STEP 2 -- the power budget (Lecture 06, slide 7)
# ===========================================================================

@dataclass
class Load:
    """One line of the power budget.

    Attributes:
        name: what the load is.
        peak_W: power drawn while it is on [W].
        duty_cycle: fraction of the time it is on, 0..1 [-].
        inside_warm_box: True if its heat is dissipated inside the insulated
            electronics box (matters for the thermal hot case; Lecture 08,
            slide 16). Drive motors are outside the box.
        rf_out_W: power that leaves as radio waves instead of heat [W]
            (only the transmitter has this; ~2 W of 12 W in Lecture 08).
    """
    name: str
    peak_W: float
    duty_cycle: float
    inside_warm_box: bool = True
    rf_out_W: float = 0.0

    @property
    def average_W(self) -> float:
        """Average power = peak x duty cycle [W]."""
        return self.peak_W * self.duty_cycle

    @property
    def heat_in_box_when_on_W(self) -> float:
        """Heat this load puts into the warm box while it is on [W]."""
        if not self.inside_warm_box:
            return 0.0
        return self.peak_W - self.rf_out_W


def power_budget(loads: List[Load], design_margin: float) -> Dict[str, float]:
    """Sum a power budget and apply the design margin (Lecture 06 method, step 1).

    Args:
        loads: list of Load objects.
        design_margin: fractional margin on the average, e.g. 0.30 for 30 %.
    Returns:
        dict with 'average_W', 'average_with_margin_W', 'energy_per_day_Wh'.
    """
    average_W = sum(load.average_W for load in loads)
    average_with_margin_W = average_W * (1.0 + design_margin)
    energy_per_earth_day_Wh = average_with_margin_W * 24.0   # 24 h per Earth-day
    return {
        "average_W": average_W,
        "average_with_margin_W": average_with_margin_W,
        "energy_per_day_Wh": energy_per_earth_day_Wh,
    }


# ===========================================================================
# STEP 3 -- the solar array (Lecture 06, slides 10-11)
# ===========================================================================

def effective_array_efficiency(cell_efficiency: float, packing_factor: float,
                               dust_degradation_factor: float,
                               hot_cell_factor: float) -> float:
    """The derate chain WITHOUT the sun-angle term (Lecture 06, slide 10).

    The sun-angle term (cos theta) is handled separately by
    array_incidence_cosine(), because it changes through the day.

    Args:
        cell_efficiency: nameplate cell efficiency at ~28 deg C [-] (e.g. 0.30).
        packing_factor: fraction of panel area covered by cells [-] (~0.85).
        dust_degradation_factor: dust + assembly losses [-] (~0.85).
        hot_cell_factor: loss from running hot [-] (~0.85 at +100 deg C).
    Returns:
        effective efficiency at normal incidence [-].
    """
    return cell_efficiency * packing_factor * dust_degradation_factor * hot_cell_factor


def array_incidence_cosine(sun_elevation: float, tilt_deg: float,
                           pointing: str, morning: bool) -> float:
    """cos(theta) between the Sun and the array normal.

    The Sun moves in the east-west plane at the equator, so the geometry
    reduces to angles in that plane.

    Args:
        sun_elevation: Sun elevation e [deg].
        tilt_deg: array tilt from horizontal, beta [deg]. 0 = flat on the roof.
        pointing: one of
            'horizontal'   -- flat panel; cos theta = sin e (tilt ignored).
            'fixed_east'   -- tilted toward the east: good in the morning,
                              poor in the afternoon.
            'fixed_west'   -- the mirror image.
            'parked_to_sun'-- the rover yaws when parked so the tilt faces
                              the Sun's azimuth (costs operations time,
                              not a mechanism).
        morning: True if the Sun is in the east (before local noon).
    Returns:
        cos(theta), clipped at 0 (no power if the Sun is behind the panel).
    """
    if pointing == "horizontal":
        return max(0.0, math.sin(math.radians(sun_elevation)))

    if pointing == "parked_to_sun":
        tilt_faces_sun = True
    elif pointing == "fixed_east":
        tilt_faces_sun = morning
    elif pointing == "fixed_west":
        tilt_faces_sun = not morning
    else:
        raise ValueError(f"Unknown array pointing '{pointing}'")

    # Normal of a panel tilted by beta toward the Sun's side has elevation
    # (90 - beta) on the Sun's side: the angle to the Sun is 90 - e - beta,
    # so cos(theta) = sin(e + beta). Tilted away: cos(theta) = sin(e - beta).
    signed_tilt_deg = tilt_deg if tilt_faces_sun else -tilt_deg
    return max(0.0, math.sin(math.radians(sun_elevation + signed_tilt_deg)))


def array_power_W(area_m2: float, eta_effective: float, cos_theta: float) -> float:
    """Array output P = S x eta_eff x cos(theta) x A [W]."""
    return SOLAR_FLUX_W_m2 * eta_effective * cos_theta * area_m2


def size_array_m2(target_power_W: float, eta_effective: float,
                  cos_theta_worst: float) -> float:
    """Array area to deliver target_power_W at the worst-case sun angle.

    A = P_target / (S x eta_eff x cos(theta_worst))   (Lecture 06, WE1)
    """
    if cos_theta_worst <= 0.0:
        raise ValueError("Sun is behind the array at the worst case -- "
                         "the array cannot be sized for this ConOps.")
    return target_power_W / (SOLAR_FLUX_W_m2 * eta_effective * cos_theta_worst)


# ===========================================================================
# STEP 4 -- the battery (Lecture 06, slides 18-20)
# ===========================================================================

def size_battery_Wh(peak_load_W: float, array_output_during_peak_W: float,
                    peak_duration_h: float, depth_of_discharge: float,
                    capacity_margin: float) -> Dict[str, float]:
    """Battery capacity for the worst sustained peak deficit (Lecture 06, WE2).

    Args:
        peak_load_W: total load during the worst peak (e.g. driving) [W].
        array_output_during_peak_W: what the array delivers at that time [W].
        peak_duration_h: how long the peak lasts [h].
        depth_of_discharge: usable fraction of capacity, 0..1 [-].
        capacity_margin: extra fractional margin on top [-].
    Returns:
        dict with 'deficit_W', 'energy_deficit_Wh', 'capacity_Wh'.
    """
    deficit_W = max(0.0, peak_load_W - array_output_during_peak_W)
    energy_deficit_Wh = deficit_W * peak_duration_h
    capacity_Wh = energy_deficit_Wh / depth_of_discharge * (1.0 + capacity_margin)
    return {"deficit_W": deficit_W,
            "energy_deficit_Wh": energy_deficit_Wh,
            "capacity_Wh": capacity_Wh}


# ===========================================================================
# STEP 5 -- the radiator in the HOT case (Lecture 08, WE1 and WE2)
# ===========================================================================

def radiator_net_flux_W_m2(radiator_temp_K: float, emittance: float,
                           absorptance: float, sun_elevation: float,
                           orientation: str) -> Dict[str, float]:
    """Net heat a radiator rejects per square metre (Lecture 08, slide 13).

    q_net = emitted - absorbed (sunlight, albedo and ground IR).

    'up'      : sees the Sun (alpha*S*sin e) and no ground (F = 0).
    'vertical': at the equator a N/S-facing wall never sees the Sun directly;
                it sees half ground (F = 0.5), so it absorbs albedo
                (alpha*a*S*sin e) and ground IR (eps*(1-a)*S*sin e).

    Args:
        radiator_temp_K: radiator temperature [K]  <-- KELVIN.
        emittance: IR emittance epsilon [-].
        absorptance: solar absorptance alpha [-] (DUSTY/EOL for the hot case,
            CLEAN/BOL for the cold case).
        sun_elevation: Sun elevation e [deg].
        orientation: 'up' or 'vertical'.
    Returns:
        dict with 'emitted_W_m2', 'absorbed_W_m2', 'net_W_m2'.
    """
    if radiator_temp_K < 100.0:
        raise ValueError("Radiator temperature looks like deg C, not kelvin. "
                         "Use c_to_k().")
    sin_e = max(0.0, math.sin(math.radians(sun_elevation)))
    emitted_W_m2 = emittance * SIGMA_W_m2K4 * radiator_temp_K ** 4

    if orientation == "up":
        absorbed_W_m2 = absorptance * SOLAR_FLUX_W_m2 * sin_e
    elif orientation == "vertical":
        view_factor_ground = 0.5
        albedo_W_m2 = absorptance * LUNAR_ALBEDO * SOLAR_FLUX_W_m2 * sin_e
        ground_ir_W_m2 = emittance * (1.0 - LUNAR_ALBEDO) * SOLAR_FLUX_W_m2 * sin_e
        absorbed_W_m2 = view_factor_ground * (albedo_W_m2 + ground_ir_W_m2)
    else:
        raise ValueError(f"Unknown radiator orientation '{orientation}'")

    return {"emitted_W_m2": emitted_W_m2,
            "absorbed_W_m2": absorbed_W_m2,
            "net_W_m2": emitted_W_m2 - absorbed_W_m2}


def radiator_equilibrium_temp_K(radiator_area_m2: float, heat_in_W: float,
                                emittance: float, absorptance: float,
                                sun_elevation: float) -> float:
    """Single-node steady state of an UP-facing radiator (Lecture 08, slide 37).

    Solves  eps*sigma*A*T^4 = Q_in + alpha*S*sin(e)*A  for T.

    Use it to PREDICT the radiator temperature for your thermal budget table
    once the area is fixed (sizing goes the other way: fix T, find A).

    Returns:
        radiator temperature [K].
    """
    sin_e = max(0.0, math.sin(math.radians(sun_elevation)))
    absorbed_W = absorptance * SOLAR_FLUX_W_m2 * sin_e * radiator_area_m2
    return ((heat_in_W + absorbed_W)
            / (emittance * SIGMA_W_m2K4 * radiator_area_m2)) ** 0.25


def pcm_mass_kg(peak_excess_W: float, peak_duration_h: float,
                latent_heat_kJ_kg: float = 200.0) -> float:
    """Phase-change material to absorb a peak (Lecture 08, WE4).

    m = (excess power x duration) / latent heat.

    Args:
        peak_excess_W: power above what the radiator rejects [W].
        peak_duration_h: duration of the peak [h].
        latent_heat_kJ_kg: latent heat of the wax [kJ/kg] (paraffin ~200).
    Returns:
        wax mass [kg] (add a container).
    """
    energy_kJ = peak_excess_W * peak_duration_h * 3600.0 / 1000.0
    return energy_kJ / latent_heat_kJ_kg


def mli_leak_W(effective_emittance: float, mli_area_m2: float,
               outer_temp_K: float, inner_temp_K: float) -> float:
    """Heat through an MLI blanket: Q = eps* sigma A (T_out^4 - T_in^4) [W].

    Positive = heat leaking INTO the box. Both temperatures in KELVIN.
    (Lecture 08, slide 31.)
    """
    return (effective_emittance * SIGMA_W_m2K4 * mli_area_m2
            * (outer_temp_K ** 4 - inner_temp_K ** 4))


def size_radiator_m2(heat_to_reject_W: float, net_flux_W_m2: float,
                     area_margin: float) -> Dict[str, float]:
    """Radiator area A = Q / q_net, then x (1 + margin) (Lecture 08, WE2)."""
    if net_flux_W_m2 <= 0.0:
        raise ValueError("Net flux is <= 0: this radiator ABSORBS more than it "
                         "emits (it is a heater). Change orientation or temperature.")
    area_m2 = heat_to_reject_W / net_flux_W_m2
    return {"area_m2": area_m2, "area_with_margin_m2": area_m2 * (1.0 + area_margin)}


# ===========================================================================
# STEP 6 -- the heater in the COLD case (Lecture 08, WE3)
# ===========================================================================

def cold_case_heater_W(radiator_area_m2: float, radiator_temp_K: float,
                       emittance_effective: float, absorptance_clean: float,
                       sun_elevation: float, mli_loss_W: float,
                       minimum_dissipation_W: float) -> Dict[str, float]:
    """Heater power to hold the box warm at the cold case (Lecture 08, WE3).

    heater = (radiator loss + MLI loss) - minimum internal dissipation

    Use the CLEAN (beginning-of-life) absorptance here: a clean radiator
    absorbs less sunlight, so it runs colder -- that is the worst case.
    For louvres, pass the closed-louvre effective emittance (~0.15).

    Returns:
        dict with intermediate steps and 'heater_W' (>= 0).
    """
    flux = radiator_net_flux_W_m2(radiator_temp_K, emittance_effective,
                                  absorptance_clean, sun_elevation, "up")
    radiator_loss_W = flux["net_W_m2"] * radiator_area_m2
    total_loss_W = radiator_loss_W + mli_loss_W
    heater_W = max(0.0, total_loss_W - minimum_dissipation_W)
    return {"emitted_W_m2": flux["emitted_W_m2"],
            "sun_absorbed_W_m2": flux["absorbed_W_m2"],
            "radiator_loss_W": radiator_loss_W,
            "total_loss_W": total_loss_W,
            "heater_W": heater_W}


def battery_zone_heater_W(mli_effective_emittance: float, zone_mli_area_m2: float,
                          battery_temp_K: float, surroundings_temp_K: float,
                          mount_conductance_WK: float, mount_temp_K: float) -> float:
    """Heater to keep an isolated battery zone warm (Lecture 08, slide 40).

    Loss = radiation through the zone's MLI + conduction through the
    isolating standoffs. All temperatures in KELVIN.
    """
    radiative_W = mli_leak_W(mli_effective_emittance, zone_mli_area_m2,
                             battery_temp_K, surroundings_temp_K)
    conductive_W = mount_conductance_WK * (battery_temp_K - mount_temp_K)
    return radiative_W + conductive_W


def heater_rating_at_nominal_bus_W(required_W: float, bus_min_V: float,
                                   bus_nominal_V: float) -> float:
    """A resistive heater's power falls as V^2 (Lecture 08, slide 32).

    To deliver required_W at the MINIMUM bus voltage, it must be rated at
    required_W x (V_nom / V_min)^2 at the nominal voltage.
    """
    return required_W * (bus_nominal_V / bus_min_V) ** 2


# ===========================================================================
# Your design, all inputs in one place
# ===========================================================================

@dataclass
class RoverDesign:
    """All design inputs for A03. Change these to YOUR values.

    Defaults reproduce the Lecture 06 / Lecture 08 reference numbers.
    """
    # --- Step 1: ConOps window, Earth-days relative to local noon ----------
    mission_start_from_noon_days: float = -5.0     # landing / first operation
    mission_end_from_noon_days: float = +5.0       # end of mission (10 days)
    first_driving_from_noon_days: float = -5.0     # when driving starts
    siesta_half_width_days: float = 0.0            # noon standby +/- this (0 = none);
                                                   # 11:00-13:00 local = 1.23 days

    # --- Step 2: loads ------------------------------------------------------
    loads: List[Load] = field(default_factory=lambda: [
        Load("Avionics ideal / C&DH", 2.5, 1, inside_warm_box=True),
        Load("Mobility (driving)", 7.45, 0.4167, inside_warm_box=False),
        Load("Comms (downlink)", 7.4, 0.17375, inside_warm_box=True, rf_out_W=2.0),
        Load("Payload (imaging)", 6.225, 0.111, inside_warm_box=True),
        Load("Heaters", 0.0, 0.20, inside_warm_box=True),
    ])
    power_design_margin: float = 0.30              # 30 % on the average

    # --- Step 3: solar array --------------------------------------------------
    cell_efficiency: float = 0.30
    packing_factor: float = 0.85
    dust_degradation_factor: float = 0.9#0.85
    hot_cell_factor: float = 0.85
    array_pointing: str = "horizontal"             # see array_incidence_cosine
    array_tilt_deg: float = 0.0
    array_on_roof: bool = False                     # True if it uses the top face
    array_design_power_W: float = 35.0             # target at the worst sun angle
    array_specific_mass_kg_m2: float = 2.546#2.5         # rigid panel incl. substrate

    # --- Step 4: battery ------------------------------------------------------
    driving_peak_load_W: float = 25.0              # mobility + avionics while driving
    driving_session_h: float = 10.0
    shadow_event_h: float = 0.5                    # driving into a crater's shadow:
                                                   # array gives 0 W for this long
    depth_of_discharge: float = 0.70
    battery_capacity_margin: float = 0.20
    battery_specific_energy_Wh_kg: float = 250.0   # pack level, not cell level

    # --- Step 5: radiator, hot case -----------------------------------------
    radiator_orientation: str = "up"
    radiator_emittance: float = 0.80               # OSR
    radiator_alpha_dusty: float = 0.15             # EOL, hot case
    radiator_alpha_clean: float = 0.08             # BOL, cold case
    radiator_hot_temp_degC: float = 24.0
    # How the box dissipation is counted while OPERATING in the hot period:
    #   'peak'    -- every in-box load on at once (Lecture 08 WE2: 19 W)
    #   'average' -- duty-averaged over a day, with PCM sized for the largest
    #                single peak (Lecture 08 WE4). Must be justified in the ConOps.
    hot_ops_dissipation_mode: str = "peak"
    siesta_dissipation_in_box_W: float = 7.4       # avionics only, during siesta
    pcm_peak_duration_h: float = 4.3333               # longest single peak (comms pass)
    plate_gradient_K: float = 3.8                 # electronics above radiator
    electronics_max_degC: float = 35.0             # T-1 from A1
    mli_effective_emittance: float = 0.03
    mli_area_m2: float = 0.28
    box_outer_hot_temp_degC: float = 77.41         # MLI outer cover at noon
    radiator_area_margin: float = 0.20
    radiator_specific_mass_kg_m2: float = 3.0      # plate + OSR tiles

    # --- Step 6: heater, cold case ---------------------------------------------
    cold_case_sun_elevation_deg: float = 25.0      # dawn
    radiator_cold_temp_degC: float =-75.9
    cold_mli_loss_W: float = -1.25
    standby_dissipation_W: float = 2.5
    radiator_area_used_m2: float = 0.0             # 0 = use the sized radiator
    louvre_closed_emittance: float = 0.0           # 0 = no louvres
    bus_nominal_V: float = 18.0
    bus_min_V: float = 5.0

    # Battery zone (Lecture 08, slide 40 reference: ~1.1 W)
    battery_zone_mli_area_m2: float = 0.05
    battery_zone_conductance_WK: float = 0.01
    battery_min_temp_degC: float = 0.0
    battery_zone_surroundings_degC: float = -80.0

    # --- Step 7: integration ------------------------------------------------------
    roof_area_m2: float = 0.1#0.232-0.0033                     # 0.5 m x 0.5 m updated for our case
    thermal_uncertainty_margin_K: float = 10.0     # your margin policy (ECSS ~10 K)


# ===========================================================================
# Running the whole chain
# ===========================================================================

def run_design(d: RoverDesign, verbose: bool = True) -> Dict[str, float]:
    """Run Steps 1-7 on a RoverDesign and return the key results.

    Prints every intermediate value when verbose=True so you can compare it
    with your own hand calculation.
    """
    out: Dict[str, float] = {}
    flags: List[str] = []

    def say(text: str = "") -> None:
        if verbose:
            print(text)

    # ---------------- Step 1 ---------------------------------------------
    say("=" * 72)
    say("STEP 1 -- ConOps window and Sun elevation")
    e_start = sun_elevation_deg(d.mission_start_from_noon_days)
    e_end = sun_elevation_deg(d.mission_end_from_noon_days)
    e_drive = sun_elevation_deg(d.first_driving_from_noon_days)
    e_min = min(e_start, e_end)
    say(f"  Sun rate                     = {SUN_RATE_deg_per_day:.2f} deg/Earth-day")
    say(f"  Elevation at mission start   = {e_start:6.1f} deg")
    say(f"  Elevation at mission end     = {e_end:6.1f} deg")
    say(f"  Elevation when driving starts= {e_drive:6.1f} deg")
    say(f"  -> lowest Sun you must work with: {e_min:.1f} deg")
    if e_min <= 0:
        flags.append("ConOps window extends past sunrise/sunset -- the rover "
                     "would be in the dark.")
    duration_days = d.mission_end_from_noon_days - d.mission_start_from_noon_days
    say(f"  Mission duration             = {duration_days:.1f} Earth-days")
    out.update(e_min_deg=e_min, e_start_deg=e_start, e_end_deg=e_end,
               e_drive_deg=e_drive)

    # ---------------- Step 2 ---------------------------------------------
    say("\nSTEP 2 -- Power budget (Lecture 06 method)")
    say(f"  {'Load':<22}{'Peak W':>8}{'Duty':>8}{'Avg W':>8}")
    for load in d.loads:
        say(f"  {load.name:<22}{load.peak_W:8.1f}{load.duty_cycle:8.2f}"
            f"{load.average_W:8.2f}")
    budget = power_budget(d.loads, d.power_design_margin)
    say(f"  Average load                 = {budget['average_W']:.2f} W")
    say(f"  x (1 + {d.power_design_margin:.2f}) margin        = "
        f"{budget['average_with_margin_W']:.2f} W")
    say(f"  Energy per Earth-day         = {budget['energy_per_day_Wh']:.0f} Wh")
    out.update(budget)

    # ---------------- Step 3 ---------------------------------------------
    say("\nSTEP 3 -- Solar array (Lecture 06 derate chain)")
    eta_eff = effective_array_efficiency(d.cell_efficiency, d.packing_factor,
                                         d.dust_degradation_factor, d.hot_cell_factor)
    say(f"  eta_eff at normal incidence  = {d.cell_efficiency:.2f} x "
        f"{d.packing_factor:.2f} x {d.dust_degradation_factor:.2f} x "
        f"{d.hot_cell_factor:.2f} = {eta_eff:.3f}")

    # Worst cos(theta) over the operating window (sample every 0.05 day)
    worst_cos, worst_t = 1.0, 0.0
    t = d.mission_start_from_noon_days
    while t <= d.mission_end_from_noon_days + 1e-9:
        cos_t = array_incidence_cosine(sun_elevation_deg(t), d.array_tilt_deg,
                                       d.array_pointing, morning=(t < 0))
        if cos_t < worst_cos:
            worst_cos, worst_t = cos_t, t
        t += 0.05
    say(f"  Pointing: {d.array_pointing}, tilt {d.array_tilt_deg:.0f} deg")
    say(f"  Worst cos(theta) in window   = {worst_cos:.3f} "
        f"(at {worst_t:+.2f} days, e = {sun_elevation_deg(worst_t):.1f} deg)")
    say(f"  [Lecture 06 WE1 assumed cos(theta) = 0.75, i.e. a Sun elevation of "
        f"~{math.degrees(math.asin(0.75)):.0f} deg for a flat array]")
    area_array_m2 = size_array_m2(d.array_design_power_W, eta_eff, worst_cos)
    say(f"  Area for {d.array_design_power_W:.1f} W at worst case  = "
        f"{d.array_design_power_W:.1f} / (1361 x {eta_eff:.3f} x {worst_cos:.3f})"
        f" = {area_array_m2:.3f} m^2")
    p_noon_W = array_power_W(area_array_m2, eta_eff,
                             array_incidence_cosine(90.0, d.array_tilt_deg,
                                                    d.array_pointing, True))
    say(f"  Same array at noon delivers  = {p_noon_W:.1f} W")
    out.update(eta_eff=eta_eff, worst_cos=worst_cos, array_area_m2=area_array_m2,
               array_noon_W=p_noon_W,
               array_mass_kg=area_array_m2 * d.array_specific_mass_kg_m2)

    # ---------------- Step 4 ---------------------------------------------
    say("\nSTEP 4 -- Battery (Lecture 06 WE2 method)")
    cos_drive = array_incidence_cosine(e_drive, d.array_tilt_deg,
                                       d.array_pointing,
                                       morning=(d.first_driving_from_noon_days < 0))
    p_array_drive_W = array_power_W(area_array_m2, eta_eff, cos_drive)
    batt_lowsun = size_battery_Wh(d.driving_peak_load_W, p_array_drive_W,
                                  d.driving_session_h, d.depth_of_discharge,
                                  d.battery_capacity_margin)
    batt_shadow = size_battery_Wh(d.driving_peak_load_W, 0.0, d.shadow_event_h,
                                  d.depth_of_discharge, d.battery_capacity_margin)
    say(f"  Case 1, low-Sun drive: array at e = {e_drive:.0f} deg gives "
        f"{p_array_drive_W:.1f} W; deficit {batt_lowsun['deficit_W']:.1f} W x "
        f"{d.driving_session_h:.1f} h = {batt_lowsun['energy_deficit_Wh']:.1f} Wh")
    say(f"  Case 2, shadow drive:  array 0 W; deficit {d.driving_peak_load_W:.0f} W x "
        f"{d.shadow_event_h:.1f} h = {batt_shadow['energy_deficit_Wh']:.1f} Wh")
    batt = max(batt_lowsun, batt_shadow, key=lambda b: b["capacity_Wh"])
    say(f"  Governing: Capacity = {batt['energy_deficit_Wh']:.1f} / "
        f"{d.depth_of_discharge:.2f} x {1 + d.battery_capacity_margin:.2f} "
        f"= {batt['capacity_Wh']:.1f} Wh")
    batt_mass_kg = batt["capacity_Wh"] / d.battery_specific_energy_Wh_kg
    say(f"  Battery mass at {d.battery_specific_energy_Wh_kg:.0f} Wh/kg "
        f"= {batt_mass_kg:.2f} kg")
    out.update(battery_Wh=batt["capacity_Wh"], battery_kg=batt_mass_kg,
               array_at_drive_W=p_array_drive_W)

    # ---------------- Step 5 ---------------------------------------------
    say("\nSTEP 5 -- Radiator, HOT cases (Lecture 08 WE1, WE2, WE4)")
    # In-box loads, heaters excluded (they are off in the hot case)
    box_loads = [l for l in d.loads
                 if l.inside_warm_box and "heater" not in l.name.lower()]
    peak_in_box_W = 7.4 #sum(l.heat_in_box_when_on_W for l in box_loads)
    avg_in_box_W = sum(l.heat_in_box_when_on_W * l.duty_cycle for l in box_loads)
    if d.hot_ops_dissipation_mode == "peak":
        ops_in_box_W, pcm_kg = peak_in_box_W, 0.0
    elif d.hot_ops_dissipation_mode == "average":
        ops_in_box_W = avg_in_box_W
        biggest_peak_W = max((l.heat_in_box_when_on_W for l in box_loads),
                             default=0.0)
        pcm_kg = pcm_mass_kg(biggest_peak_W, d.pcm_peak_duration_h)
    else:
        raise ValueError("hot_ops_dissipation_mode must be 'peak' or 'average'")
    say(f"  In-box heat: all-on peak {peak_in_box_W:.1f} W; duty-averaged "
        f"{avg_in_box_W:.1f} W; using '{d.hot_ops_dissipation_mode}' "
        f"-> {ops_in_box_W:.1f} W while operating")

    window_has_noon = (d.mission_start_from_noon_days <= 0
                       <= d.mission_end_from_noon_days)
    e_noon = 90.0 if window_has_noon else max(e_start, e_end)
    t_rad_hot_K = c_to_k(d.radiator_hot_temp_degC)
    leak_hot_W = mli_leak_W(d.mli_effective_emittance, d.mli_area_m2,
                            c_to_k(d.box_outer_hot_temp_degC), t_rad_hot_K)
    say(f"  MLI leak in (outer {d.box_outer_hot_temp_degC:+.0f} degC) = "
        f"{leak_hot_W:.1f} W")

    # Case H1: operating at the highest Sun the rover operates in
    if d.siesta_half_width_days > 0 and window_has_noon:
        e_ops = sun_elevation_deg(d.siesta_half_width_days)
        hot_cases = [("H1 operating at siesta edge", e_ops, ops_in_box_W),
                     ("H2 siesta at noon", e_noon, d.siesta_dissipation_in_box_W)]
    else:
        hot_cases = [("H1 operating at noon", e_noon, ops_in_box_W)]

    sized = []
    for label, e_case, q_box_W in hot_cases:
        flux = radiator_net_flux_W_m2(t_rad_hot_K, d.radiator_emittance,
                                      d.radiator_alpha_dusty, e_case,
                                      d.radiator_orientation)
        q_W = q_box_W + leak_hot_W
        #print(q_W)
        #print(leak_hot_W)
        rad_case = size_radiator_m2(q_W, flux["net_W_m2"], d.radiator_area_margin)
        say(f"  {label}: e = {e_case:.1f} deg; emitted {flux['emitted_W_m2']:.0f}"
            f" - absorbed {flux['absorbed_W_m2']:.0f} = {flux['net_W_m2']:.0f} W/m^2;"
            f" Q = {q_box_W:.1f} + {leak_hot_W:.1f} = {q_W:.1f} W -> "
            f"{rad_case['area_m2']:.3f} x{1 + d.radiator_area_margin:.2f} = "
            f"{rad_case['area_with_margin_m2']:.3f} m^2")
        sized.append((rad_case["area_with_margin_m2"], label, e_case, q_W,
                      flux["net_W_m2"]))
    area_rad_m2, gov_label, e_gov, q_gov_W, net_gov = max(sized)
    say(f"  Governing: {gov_label} -> radiator {area_rad_m2:.3f} m^2")
    if pcm_kg > 0:
        say(f"  PCM to absorb the largest single peak for "
            f"{d.pcm_peak_duration_h:.1f} h = {pcm_kg:.2f} kg wax (+ container)")
        


    # Predicted temperatures with the chosen area, for the thermal budget table
    t_rad_pred_K = radiator_equilibrium_temp_K(area_rad_m2, q_gov_W,
                                               d.radiator_emittance,
                                               d.radiator_alpha_dusty, e_gov)
    t_elec_hot_degC = k_to_c(t_rad_pred_K) + d.plate_gradient_K
    hot_margin_K = d.electronics_max_degC - t_elec_hot_degC
    say(f"  Predicted radiator {k_to_c(t_rad_pred_K):+.1f} degC -> electronics "
        f"{t_elec_hot_degC:+.1f} degC; margin to {d.electronics_max_degC:+.0f} degC"
        f" = {hot_margin_K:.1f} K (policy {d.thermal_uncertainty_margin_K:.0f} K)")
    out.update(rad_net_hot_W_m2=net_gov, q_hot_W=q_gov_W,
               radiator_area_m2=area_rad_m2, pcm_kg=pcm_kg,
               electronics_hot_degC=t_elec_hot_degC, hot_margin_K=hot_margin_K,
               radiator_mass_kg=area_rad_m2 * d.radiator_specific_mass_kg_m2)
    if hot_margin_K < d.thermal_uncertainty_margin_K:
        flags.append("Hot-case electronics margin is below your margin policy.")
        
    rad = {"area_with_margin_m2": area_rad_m2}

    # ---------------- Step 6 ---------------------------------------------
    say("\nSTEP 6 -- Heater, COLD case (Lecture 08 WE3)")
    area_cold_m2 = d.radiator_area_used_m2 or rad["area_with_margin_m2"]
    eps_cold = d.louvre_closed_emittance or d.radiator_emittance
    cold = cold_case_heater_W(area_cold_m2, c_to_k(d.radiator_cold_temp_degC),
                              eps_cold, d.radiator_alpha_clean,
                              d.cold_case_sun_elevation_deg, d.cold_mli_loss_W,
                              d.standby_dissipation_W)
    say(f"  Sun e = {d.cold_case_sun_elevation_deg:.0f} deg; radiator "
        f"{d.radiator_cold_temp_degC:+.0f} degC; area {area_cold_m2:.3f} m^2; "
        f"eps {eps_cold:.2f}; CLEAN alpha {d.radiator_alpha_clean:.2f}")
    say(f"  Emitted {cold['emitted_W_m2']:.0f} - Sun {cold['sun_absorbed_W_m2']:.1f}"
        f" W/m^2 -> radiator loss {cold['radiator_loss_W']:.1f} W")
    say(f"  + MLI {d.cold_mli_loss_W:.1f} W = {cold['total_loss_W']:.1f} W; "
        f"- standby {d.standby_dissipation_W:.1f} W -> heater "
        f"{cold['heater_W']:.1f} W")
    rated_W = heater_rating_at_nominal_bus_W(cold["heater_W"], d.bus_min_V,
                                             d.bus_nominal_V)
    say(f"  Rated at {d.bus_nominal_V:.0f} V to deliver it at {d.bus_min_V:.0f} V"
        f" = {rated_W:.1f} W")
    zone_W = battery_zone_heater_W(d.mli_effective_emittance,
                                   d.battery_zone_mli_area_m2,
                                   c_to_k(d.battery_min_temp_degC),
                                   c_to_k(d.battery_zone_surroundings_degC),
                                   d.battery_zone_conductance_WK,
                                   c_to_k(d.battery_zone_surroundings_degC))
    say(f"  Battery-zone-only heater     = {zone_W:.2f} W "
        f"(alternative: let the box cool, keep only the battery warm)")
    out.update(heater_W=cold["heater_W"], heater_rated_W=rated_W,
               battery_zone_heater_W=zone_W)

    # ---------------- Step 7 ---------------------------------------------
    #say(f"hot_margin_K {hot_margin_K:.1f} and d.thermal_uncertainty_margin_K{d.thermal_uncertainty_margin_K:.1f} ")
    say("\nSTEP 7 -- Integration checks (where power meets thermal)")
    # (a) Roof
    rad_area = rad["area_with_margin_m2"] if d.radiator_orientation == "up" else 0.0
    roof_used_m2 = rad["area_with_margin_m2"] if d.radiator_orientation == "up" else 0.0
    if d.array_on_roof:
        roof_used_m2 += area_array_m2/2 #check here Area solar panel half the size
        #say(f"area of array is: {area_array_m2:.4f}m^2 and rad is: {rad_area:.4f}m^2 ")
    roof_ok = roof_used_m2 <= d.roof_area_m2
    say(f"  (a) Roof: {roof_used_m2:.3f} of {d.roof_area_m2:.3f} m^2 used -> "
        f"{'OK' if roof_ok else 'CONFLICT'}")
    if not roof_ok:
        flags.append("Array + radiator do not fit on the roof (Lecture 08 slide 24).")

    # (b) Dawn energy balance: array at the cold-case Sun vs standby + heater
    cos_cold = array_incidence_cosine(d.cold_case_sun_elevation_deg,
                                      d.array_tilt_deg, d.array_pointing, True)
    p_array_cold_W = array_power_W(area_array_m2, eta_eff, cos_cold)
    dawn_load_W = d.standby_dissipation_W + rated_W
    dawn_ok = p_array_cold_W >= dawn_load_W
    #say(f"dawn load W {dawn_load_W: .2f} W d.standby_dissipation_W: {d.standby_dissipation_W:.2f} and rated W{rated_W:.2f} ")
    #say(f"p_array_cold_W {p_array_cold_W:.2f} W")

    say(f"  (b) Dawn: array {p_array_cold_W:.1f} W vs standby + heater "
        f"{dawn_load_W:.1f} W -> {'OK' if dawn_ok else 'DOES NOT CLOSE'}")
    if not dawn_ok:
        flags.append("Dawn energy balance does not close: the battery drains "
                     "for as long as the cold case lasts (days, not hours).")
    out.update(array_at_dawn_W=p_array_cold_W, dawn_load_W=dawn_load_W)

    # (b2) Array carries the average over the day at the worst Sun
    p_array_worst_W = array_power_W(area_array_m2, eta_eff, worst_cos)
    say(f"  (b2) Worst-Sun array {p_array_worst_W:.1f} W vs average load with "
        f"margin {budget['average_with_margin_W']:.1f} W -> "
        f"{'OK' if p_array_worst_W >= budget['average_with_margin_W'] else 'SHORT'}")
    if p_array_worst_W < budget["average_with_margin_W"]:
        flags.append("At the worst Sun the array cannot carry the average load.")
    if d.cold_case_sun_elevation_deg > e_min + 0.5:
        flags.append(f"Cold case uses e = {d.cold_case_sun_elevation_deg:.0f} deg but "
                     f"the ConOps reaches e = {e_min:.0f} deg: the cold case is not "
                     f"the coldest point of your mission.")

    # (c) Heater fed back into the power budget
    heater_lines = [l for l in d.loads if "heater" in l.name.lower()]
    heater_budget_peak_W = max((l.peak_W for l in heater_lines), default=0.0)
    fed_back = heater_budget_peak_W + 1e-6 >= rated_W
    say(f"  (c) Heater line in power budget {heater_budget_peak_W:.1f} W vs "
        f"needed {rated_W:.1f} W -> {'OK' if fed_back else 'NOT FED BACK'}")
    if not fed_back:
        flags.append("Heater power in the power budget is smaller than the "
                     "cold-case heater: update Step 2 and re-run.")

    # (d) Power + thermal mass feeding back to the A1 budget
    mli_kg = d.mli_area_m2 * 0.75                # MLI ~0.5-1 kg/m^2 (Lecture 08)
    pcm_total_kg = pcm_kg * 1.5                  # wax + ~50 % for the container
    mass_kg = (out["array_mass_kg"] + batt_mass_kg + out["radiator_mass_kg"]
               + mli_kg + pcm_total_kg)
    say(f"  (d) Array {out['array_mass_kg']:.2f} + battery {batt_mass_kg:.2f} + "
        f"radiator {out['radiator_mass_kg']:.2f} + MLI {mli_kg:.2f} + PCM "
        f"{pcm_total_kg:.2f} = {mass_kg:.2f} kg (add PCDU, harness, heaters yourself)")
    out["power_thermal_mass_kg"] = mass_kg

    say("\nFLAGS")
    if flags:
        for f_ in flags:
            say(f"  ! {f_}")
    else:
        say("  none -- now check that every number agrees with your report")
    say("=" * 72)
    out["n_flags"] = float(len(flags))
    return out


def reproduce_lecture_examples() -> None:
    """Re-run the Lecture 08 worked examples, so you can check the tool.

    Note on WE3: the slide says 'clean surfaces' but uses alpha = 0.20 (the
    DUSTY value). With the clean value alpha = 0.08 the heater is larger.
    Use the clean value -- it is the worst case for the cold case.
    """
    print("Lecture 08 checks")
    up = radiator_net_flux_W_m2(c_to_k(45), 0.80, 0.20, 90, "up")
    vert = radiator_net_flux_W_m2(c_to_k(45), 0.80, 0.20, 90, "vertical")
    print(f"  WE1 up-facing net     = {up['net_W_m2']:+.0f} W/m^2   (slide: +193)")
    print(f"  WE1 vertical net      = {vert['net_W_m2']:+.0f} W/m^2   (slide: -31)")
    leak = mli_leak_W(0.03, 0.40, c_to_k(100), c_to_k(45))
    print(f"  WE2 MLI leak          = {leak:.1f} W          (slide: ~6)")
    print(f"  WE2 radiator          = {(19 + leak) / up['net_W_m2'] * 1.2:.3f} m^2"
          f"      (slide: 0.16)")
    we3_slide = cold_case_heater_W(0.07, c_to_k(-15), 0.80, 0.20, 10, 0.6, 5)
    we3_clean = cold_case_heater_W(0.07, c_to_k(-15), 0.80, 0.08, 10, 0.6, 5)
    print(f"  WE3 heater, alpha 0.20 (as on slide) = {we3_slide['heater_W']:.1f} W")
    print(f"  WE3 heater, alpha 0.08 (clean, correct) = {we3_clean['heater_W']:.1f} W")



if __name__ == "__main__":
    reproduce_lecture_examples()

    print("REFERENCE DESIGN A -- Lecture 06 / Lecture 08 numbers, flat array on the roof")
    run_design(RoverDesign())

    print("\nREFERENCE DESIGN B -- one way to close it (yours may differ)")
    run_design(RoverDesign(
        mission_start_from_noon_days=-5.,     # land at ~ e = 10 deg (dawn)
        mission_end_from_noon_days=+5,       # 10 Earth-days later
        first_driving_from_noon_days=-5,     # drive once e > ~23 deg
        array_pointing="parked_to_sun",        # yaw when parked
        array_tilt_deg=0.0,
        array_on_roof=False,                   # tilted panel frees the roof
        array_design_power_W=14.5,# what 35.5 jan suggested 14.5 what i thought at first
        siesta_half_width_days=1.23,           # 11:00-13:00 local standby
        hot_ops_dissipation_mode="average",    # peaks buffered by PCM (WE4)
        loads=[
            Load("Idle/Charging", 4.76, 0.2986,inside_warm_box=True),
            Load("Mobility (driving)", 14.02, 0.417, inside_warm_box=False),
            Load("Comms (downlink)", 9.66, 0.173, rf_out_W=2.0),
            Load("Payload (imaging)", 12.8, 0.11,inside_warm_box=True),
            #Load("charging", 2.5, 0.57),
        ],
    ))
