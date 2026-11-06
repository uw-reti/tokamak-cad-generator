#!python
# ============================================================================
# FNSF_DCLL_IB_OB_v13.py
#
# Written By: Jessica Wysocki
# Original code skeleton and plasma parameters: Monica Gehrig, Paul Humrickhouse
# Dimensions: Sunday Aduloju
# Date: 11/5/2026
#
# FULL INBOARD + OUTBOARD FNSF DCLL blanket sector (Coreform Cubit, Python).
# Every design dimension is a USER INPUT (mm) in one block near the top of
# this file; everything else is calculated:
#   - INBOARD PbLi duct COLUMNS are an input: the sector is split into that
#     many EQUAL wedges (22.5 / 5 = 4.5 deg); every duct, edge ducts included,
#     fills one wedge. Duct widths are calculated (edge ducts are narrower
#     because their outer side is the 38 mm wrapped-He band).
#   - OUTBOARD columns are an input too, with the same rule: Num_PbLi_cols_OB
#     equal wedges (22.5 / 9 = 2.5 deg), webs centred on the wedge lines, edge
#     ducts bounded by the wrapped-He band. Widths are calculated.
#   - PbLi ROWS are inputs (row 1 + "other rows" sizes); depths and heights
#     follow from the radial stack.
#   - He hole COUNTS: fixed rib between holes, as many as fit, leftover split
#     equally at both ends (arc rows between the He bands, column holes along
#     the duct length). Any RAFM ligament below Min_RAFM_ligament stops the build.
#   - IB manifold slot WIDTHS: outer side 4 mm inside the He band, inner side
#     10.16 mm from the housing wall. The housing wall is the radial line
#     through duct 2's extended outer side at the manifold rear face (same on
#     both sides; Manifold_housing_wall_angle = a number overrides it). Where a port enters a slot, only the two
#     slot corners on the port side are reduced by the slot test; the corners
#     on the housing-wall side keep Manifold_slot_fillet_radius.
#   - SiC corner radii: = duct fillet, inner surface of edge ducts = fillet - SiC.
#   - OB wrapped-He channels: each along its local first-wall normal; as many
#     as fit with He_channel_stack_wall (4 mm) of RAFM between neighbours at the
#     void front face (closest to the FW); the end walls are exactly half that.
# Geometry is exact (true lines / arcs) on the inboard; the manifold slot
# corner test and the v7 port fix are kept.
# Outputs: <Output_basename>.cub / .stp / _IB.stp / _OB.stp / _progress.log
#
# (original header follows)
#
# FNSF dual-coolant lead-lithium (DCLL) blanket sector: INBOARD + OUTBOARD
# in ONE Coreform Cubit model. Writes one .cub, the full .stp, and one
# .stp per blanket (_IB.stp, _OB.stp).
#
# Run from the Cubit Python console / script tab:
#   exec(open(r"C:\Users\jessi\Downloads\FNSF_DCLL_IB_OB_v13.py").read())
#
# UNITS: METERS. Machine axis = Z; both blankets span theta = 0 .. 360/N deg.
#
# LAYOUT OF THIS FILE (search for these banners)
#   1. SHARED INPUTS ........ plasma shape, sector count, switches, output names
#   2. LOGGING / HELPERS .... progress log, graphics redraw, OB first-wall end Z
#   3. build_inboard() ...... the whole inboard (IB) blanket:
#        USER INPUTS -> DERIVED GEOMETRY -> XY / Cubit helpers -> U channel,
#        port, body and manifold outlines -> PbLi / SiC layout -> He-hole
#        patterns -> input checks -> MAIN BUILD (phase 1 wrapped He,
#        phase 2 ducts / SiC / internal He holes)
#   4. build_outboard() ..... the whole outboard (OB) blanket:
#        USER INPUTS -> FNSF curve + normal offsets -> Cubit helpers ->
#        planes / sweep paths / corner trimmers -> rounded ducts -> wrapped-He
#        channels -> He-hole layouts -> MAIN BUILD (phase 1 body + wrapped He,
#        phase 2 validated ducts + SiC, internal He holes)
#   5. DRIVER ............... sizes the IB from the OB, runs both phases,
#        checks every volume, makes the material blocks, saves, exports
#
# Every code line carries a comment. Comments ending a line describe that
# line; a comment on its own line describes the statement right below it.
#
# GRAPHICS: Show_build_in_graphics = False (default) builds without redrawing
# the model after every step, which is much faster and uses less memory. Set
# it True only to watch a build. The model is shown again at the very end.
#
# HEIGHT / CENTERING
#   The outboard runs poloidally until its FW surface reaches R = OB_end_radius
#   (= R0). Its top/bottom end faces follow the curve normal, so its OUTER
#   (back) curve reaches higher than the FW. The inboard is a straight slab
#   spanning that FULL outboard height (Z = +/- Z of the OB back corner),
#   centered on Z = 0. It holds as
#   many full 20.12 mm wrapped-He levels as fit; the leftover height is split
#   equally top and bottom as solid RAFM.
#
# BUILD ORDER (FAST version: wrapped He FIRST, into the plain bodies)
#   1. Inboard  : RAFM body; wrapped-He network built as a 2-level unit cell
#                 (U + 2 ports), copied up in Z, united with the manifold slots
#                 into the ODD / EVEN circuits, then cut with 4 subtracts
#   2. Outboard : RAFM body + FW corners; wrapped-He channels (batched cuts)
#   3. Inboard  : PbLi ducts + SiC, internal He holes
#   4. Outboard : PbLi ducts + SiC, internal He holes
#   5. Blocks MAT_RAFM / MAT_HE / MAT_PBLI / MAT_SIC (+ groups), save, export
#
# WHY THIS IS FASTER
#   The old order cut ~711 inboard U/port tools one at a time into a RAFM that
#   already held thousands of full-height duct and hole faces; every cut had to
#   rebuild (and redraw) that whole body. Here the wrapped network goes into a
#   plain body in a handful of Booleans, and the ducts / holes follow in ~30.
#   The inboard U outline also uses fewer facets (straight legs = 1 face),
#   which cuts the face count and memory roughly in half.
# ============================================================================

import math  # math functions (sin, cos, radians, ...)
import time  # wall-clock timing for the progress log
import numpy as np  # numpy for vector math on the plasma curve
import cubit  # Coreform Cubit Python API (cubit.cmd, get_entities, ...)


# ============================================================================
# USER INPUTS
#   Every design dimension of BOTH blankets is set here, in millimetres
#   (`* mm` converts to metres, the model's unit). Everything else (duct
#   column counts, edge-duct widths, hole counts, total depths and heights,
#   manifold slot widths, ...) is CALCULATED from these numbers.
#   Default values = the reference IB slice (IB_blanket_slice.STEP).
# ============================================================================
mm = 1.0e-3  # millimetres -> metres

# --- machine / plasma (FNSF) ------------------------------------------------
Num_sectors = 16          # 360 / 16 = 22.5 deg sector (both blankets)
R0 = 4.8                  # major radius [m]
a = 1.2                   # minor radius [m]
kappa = 2.2               # elongation
tau = 0.63                # shaping term in R = R0 + a cos(xi + tau sin xi)
tSOL = 0.10               # plasma boundary -> FW surface [m] (IB FW at R0 - a - tSOL)
OB_end_radius = R0        # the outboard ends where its FW reaches this radius [m]; also sets both heights

# --- what to build / output ---------------------------------------------------
Build_inboard = True  # True = build the inboard blanket
Build_outboard = True  # True = build the outboard blanket (its geometry always sets the IB height)
Show_build_in_graphics = False  # False = no redraws (much faster, less memory); True = watch it build
Output_directory = "C:/Users/jessi/Downloads"  # folder for the .cub, .stp files and progress log
Output_basename = "FNSF_DCLL_IB_OB_v13"  # file name stem for every output
Create_blocks = True  # True = make the 4 material blocks (MAT_RAFM, MAT_HE, MAT_PBLI, MAT_SIC)
Create_OpenMC_material_groups = True  # True = also make DAGMC-style groups mat:RAFM, mat:He, mat:PbLi, mat:SiC
Export_step = True  # True = write the STEP files (full model + one per blanket)
STEP_units_metres = True  # True = the STEP files are LABELLED in metres (the model's real unit) -> true size in
                          # SolidWorks, geometry untouched. (Cubit's own STEP writer labels them mm = 1000x too small.)
Scale_cub_to_mm = True    # True = scale the finished model x1000 before saving the .cub (numbers in mm). Cubit
                          # prints "Model may be corrupted from the scaling operation" warnings; checked: harmless

# --- design rule ----------------------------------------------------------------
Min_RAFM_ligament = 2.0 * mm  # any RAFM rib / ligament thinner than this stops the build (mechanical limit)

# --- first wall / RAFM body --------------------------------------------------------
FW_corner_radius = 88.8 * mm  # first-wall / side-wall corner radius (IB and OB)

# --- wrapped He channels (U channels in the 38 mm FW / side band) -------------------
He_channel_height = 16.12 * mm      # void height (stacking direction)
He_channel_width = 30.0 * mm        # void width across the band
He_channel_wall = 4.0 * mm          # RAFM on each side of the void across the band (band = 4 + 30 + 4 = 38)
He_channel_stack_wall = 4.0 * mm    # IB: RAFM between stacked channels (2 above + 2 below each void)
                                    # (OB: same pitch along the band; end walls = half of this)
He_channel_end_fillet_radius = 6.35 * mm  # IB U-leg end fillets and the port bends into the slots

# --- PbLi ducts ------------------------------------------------------------------------
#   Rows: row 1 is behind the first wall; every row behind it uses the
#   "other rows" length / width.
#   INBOARD columns are an input: the sector is split into Num_PbLi_cols_IB
#   EQUAL wedges (every row uses the same dividing lines). Each inner duct
#   side sits half a web (PbLi_column_gap / 2) inside its wedge line; the edge
#   ducts' outer side is the wrapped-He band. All duct widths are calculated
#   (printed in the log; widths = distance between the midpoints of a duct's
#   two straight sides). 5 columns -> 4.5 deg wedges -> inner 246.39 / 224.76 mm,
#   edge 215.88 / 194.25 mm, as in the slice.
#   OUTBOARD columns are an input with the SAME rule: Num_PbLi_cols_OB equal
#   wedges, a web centred on every wedge line, edge ducts bounded by the
#   wrapped-He band (sized at each row's narrowest radius, as before).
Num_PbLi_rows_IB = 2                    # PbLi rows in the inboard
Num_PbLi_rows_OB = 4                    # PbLi rows in the outboard
Num_PbLi_cols_IB = 5                    # PbLi columns in every inboard row (equal wedges of 22.5 / N deg)
Num_PbLi_cols_OB = 9                    # PbLi columns in every outboard row (equal wedges of 22.5 / N deg)
PbLi_row1_length = 265.41 * mm          # row 1 length, first wall -> origin
PbLi_other_rows_length = 235.41 * mm    # length of every row behind row 1
PbLi_duct_fillet_radius = 50.8 * mm     # duct corner radius
PbLi_column_gap = 15.0 * mm             # RAFM web between neighbouring ducts (constant thickness)
PbLi_row_gap = 25.0 * mm                # RAFM between duct rows (holds the row-divider He holes)

# --- SiC flow-channel inserts -------------------------------------------------------------
SiC_thickness = 5.0 * mm                # SiC wall thickness, all around
SiC_gap = 2.0 * mm                      # PbLi gap between the duct wall and the SiC (sides, back, front of rows 2+)
SiC_front_gap_row1 = 4.0 * mm           # PbLi gap at the FRONT face of row 1
# SiC corner radii are not inputs: outer and inner surfaces = duct fillet,
# except the inner surface of edge ducts = duct fillet - SiC thickness.

# --- internal He holes ---------------------------------------------------------------------
# Row-divider holes (in each RAFM gap between duct rows; centred in the gap and along the arc)
Divider_He_hole_length = 19.0 * mm      # radial size (must fit PbLi_row_gap with Min_RAFM_ligament each side)
Divider_He_hole_width = 16.21 * mm      # size along the arc (at the hole's mid radius)
Divider_He_hole_fillet_radius = 3.0 * mm  # corner radius (also used for the rear holes)
Divider_He_hole_rib = 3.01 * mm         # RAFM between neighbouring holes (fixed); leftover goes to both ends
# Rear holes (behind the last duct row; positioned by the gaps below, not centred)
Rear_He_hole_length = 12.0 * mm         # radial size
Rear_He_hole_width = 14.83 * mm         # size along the arc (at the hole's mid radius)
Rear_He_hole_rib = 3.01 * mm            # RAFM between neighbouring holes (fixed); leftover goes to both ends
Rear_He_wall_thickness = 20.0 * mm     # last duct row back face -> RAFM back wall; the rear holes are CENTRED in it
                                        # (20 = 4 + 12 + 4). IB: the first manifold slot is Manifold_back_wall_to_first_slot
                                        # further back (8 mm from the holes).
# Column holes (in the webs between duct columns, centred on each web and along the duct length)
Column_He_hole_length = 9.0 * mm        # size along the web (first wall -> origin)
Column_He_hole_width = 9.0 * mm         # size across the web (must fit PbLi_column_gap with Min_RAFM_ligament each side)
Column_He_hole_fillet_radius = 3.0 * mm  # corner radius
Column_He_hole_rib = 3.0 * mm           # RAFM between neighbouring holes (fixed); leftover goes to both ends

# --- IB He manifold (behind the IB body, both sides) ------------------------------------------
#   Three slots per side. Each slot's outer side is Manifold_gap_to_wrapped_channel
#   inside the wrapped-He band; its inner side is Manifold_slot_to_housing_wall
#   from the housing inner wall. The housing inner wall is a RADIAL line (it
#   runs to the machine axis), the SAME on both sides (mirror images). By
#   default it is derived: the radial line through the point where the outer
#   side of PbLi duct 2 of the last row (the second-last duct on the right),
#   extended straight back, meets the manifold rear face (4.6582 deg here; the
#   slice has 4.6583 / 4.6587). Slot widths are therefore calculated.
Manifold_housing_wall_angle = None      # [deg] from the sector wall; None = derived as above, a number overrides it
Manifold_back_wall_to_first_slot = 4.0 * mm  # RAFM back wall -> front of the first slot
Manifold_slot_length = 61.33 * mm       # radial length of every slot
Manifold_slot_gap = 4.0 * mm            # RAFM between slots and behind the last slot
Manifold_slot_fillet_radius = 15.24 * mm  # slot corner radius (the slot test may reduce it where a port enters)
Manifold_slot_to_housing_wall = 10.16 * mm  # slot inner side -> housing inner wall
Manifold_gap_to_wrapped_channel = 4.0 * mm  # wrapped-He band -> slot outer side
Manifold_port_width = 32.0 * mm         # port mouth width (height = He_channel_height); must fit a slot
Manifold_housing_fillet_radius = 15.24 * mm  # housing inner wall <-> RAFM back wall fillet

# rear holes centred in the rear wall: RAFM on each side of them (4 mm here)
Rear_He_gap_to_ducts = 0.5 * (Rear_He_wall_thickness - Rear_He_hole_length)  # duct row -> rear holes [m]
Rear_He_gap_to_back_wall = Rear_He_gap_to_ducts  # rear holes -> RAFM back wall [m]

# ============================================================================
# END OF USER INPUTS (build settings that are not design dimensions are at
# the top of build_inboard() and build_outboard() below)
# ============================================================================


cubit_file_name = f"{Output_directory}/{Output_basename}.cub"  # full path of the Cubit model file
step_file_name = f"{Output_directory}/{Output_basename}.stp"  # full path of the full-model STEP file

# Progress log (also printed again at the end): open it in Notepad while
# Cubit works; the command window often does not redraw during a long run.
progress_log_file = f"{Output_directory}/{Output_basename}_progress.log"
_t0 = time.time()  # script start time (all log times are relative to this)
try:  # try to create / empty the progress log file
    open(progress_log_file, "w").close()  # open in write mode = start a new, empty log
except OSError:
    progress_log_file = None  # folder not writable: log to the console only


def log(msg):  # print a time-stamped message and append it to the progress log
    line = f"[{time.time() - _t0:8.1f} s] {msg}"  # "[  123.4 s] message"
    print(line, flush=True)  # print to the Cubit console immediately
    if progress_log_file:  # only if the log file could be created
        try:  # ignore file errors so logging never stops the build
            with open(progress_log_file, "a") as fh:  # append mode keeps earlier lines
                fh.write(line + "\n")  # one line per message
        except OSError:
            pass  # skip this message if the file is locked


def show():  # redraw Cubit's graphics window (only when Show_build_in_graphics is True)
    """Force Cubit to redraw the graphics window mid-script."""
    if not Show_build_in_graphics:
        return  # graphics off: do nothing (faster)
    for fn in ("flush_graphics", "update_graphics"):  # try both API names; Cubit versions differ
        f = getattr(cubit, fn, None)  # look up the function, None if this version lacks it
        if callable(f):  # call it only if it exists
            try:  # guard against API differences
                f()  # flush pending graphics updates
            except Exception:
                pass  # ignore: redraw is cosmetic
    cubit.cmd("display")  # ask Cubit to draw the current model


_ARC_OK = None   # outboard: probed once, whether 3-point arcs work


def _fw_end_Z():  # Z where the OB first wall reaches R = OB_end_radius (fallback IB half-height)
    """Z where the outboard FW surface (depth tSOL) reaches OB_end_radius."""
    bb = a * kappa  # plasma half-height b = a * kappa

    def RZ(xi):  # (R, Z) of the FW surface at poloidal angle xi [deg]
        x = math.radians(xi)  # degrees -> radians
        # outward normal of the plasma boundary (unnormalized)
        n = np.array([bb * math.cos(x), a * math.sin(x + tau * math.sin(x)) * (1 + tau * math.cos(x))])
        n = n / np.linalg.norm(n)  # make it a unit vector
        # plasma point + tSOL along the normal = FW point
        return (R0 + a * math.cos(x + tau * math.sin(x)) + tSOL * n[0], bb * math.sin(x) + tSOL * n[1])
    lo, hi = 0.0, 89.9  # bisection bracket in xi [deg] (midplane .. near the top)
    if (RZ(lo)[0] - OB_end_radius) * (RZ(hi)[0] - OB_end_radius) > 0:  # the bracket must straddle R = OB_end_radius
        raise ValueError("OB_end_radius is not reached on the FW surface.")  # stop: the FW never reaches that radius
    for _ in range(200):  # 200 halvings = machine precision
        mid = 0.5 * (lo + hi)  # bracket midpoint
        if (RZ(lo)[0] - OB_end_radius) * (RZ(mid)[0] - OB_end_radius) <= 0:  # root is in the lower half
            hi = mid  # move the upper bound down
        else:
            lo = mid  # move the lower bound up
    return RZ(0.5 * (lo + hi))[1]  # Z of the FW point at the root


Blanket_half_height = _fw_end_Z()   # replaced by the OB back-corner Z in the driver


def build_inboard():  # whole inboard script as a generator: yields after phase 1 and at the end
    # ============================================================================
    # FNSF_DCLL_IB_PATTERNED.py
    #
    # FNSF dual-coolant lead-lithium (DCLL) INBOARD blanket sector, built in
    # Coreform Cubit from user inputs. Writes one .cub and one .stp.
    #
    # Run from the Cubit Python console / script tab:
    #   exec(open(r"C:\Users\jessi\Downloads\FNSF_DCLL_IB_PATTERNED.py").read())
    #
    # UNITS: METERS throughout the script and in the Cubit model.
    #        (0.038 = 38 mm.)  OpenMC/DAGMC works in cm, so scale by 100 at
    #        DAGMC export time, not here.
    #
    # WHAT IS BUILT
    #   - ONE integrated RAFM solid (FW + side walls + webs + back + manifolds).
    #   - 10 PbLi ducts (2 rows x 5 columns), each with a hollow SiC insert,
    #     giving 10 SiC shells + 10 outer-gap PbLi + 10 core PbLi bodies.
    #   - Internal He holes (large middle row, large rear row, small web holes),
    #     each a retained full-height He fluid body.
    #   - N stacked wrapped-He U channels + 2N manifold ports + 6 manifold slots,
    #     assembled into FOUR He fluid bodies:
    #         ODD  circuit  (levels 1,3,5..): LEFT -> middle slot, RIGHT -> rear slot
    #         EVEN circuit  (levels 2,4,6..): LEFT -> rear slot,   RIGHT -> middle slot
    #         LEFT / RIGHT front manifold slots (not connected in this model)
    #   - Blocks MAT_RAFM / MAT_HE / MAT_PBLI / MAT_SIC and groups mat:<name>.
    #
    # PATTERNING (every repeated feature is made ONCE and copied)
    #   - Wrapped-He U channel: one prototype at level 1, copied up in Z.
    #   - Manifold ports: one prototype per (level parity, side) = 4 prototypes,
    #     copied up in Z by multiples of 2 x pitch.
    #   - Large middle / rear He rows: one prototype, rotate-copied about Z.
    #   - Small web He holes: one prototype per web segment, translate-copied
    #     along that web's radial direction.
    #   - PbLi ducts / SiC / manifold slots are full-height prisms (not repeated
    #     in Z), so they are built directly.
    #
    # Z (poloidal) STACK: level k occupies [k*pitch, (k+1)*pitch] with
    #   pitch = He_bottom_wall + He_void_height + He_top_wall = 20.12 mm.
    #   (Combined file: the IB height = the full OB outer-curve height; the
    #   leftover below one pitch is split top and bottom as solid RAFM.)
    # ============================================================================


    # ============================================================================
    # INBOARD BUILD SETTINGS (design dimensions are the USER INPUTS at the top)
    # ============================================================================
    # Slot corner test: a port mouth that lands on a slot corner arc makes a
    # near-tangent (bad) intersection. For the slots that receive ports the
    # corner radius is reduced to the largest one that keeps the mouth on the
    # straight slot side (Slot_mouth_clearance_to_arc before the arc), and a
    # short test piece is ACIS-validated; it steps down by Slot_radius_step if
    # the test fails.
    Slot_mouth_clearance_to_arc = 0.0005  # mouth side -> start of the corner arc [m]
    Slot_radius_step = 0.00025  # step down if a test piece fails [m]
    Slot_radius_max_steps = 6  # at most 6 smaller radii are tried
    He_port_fillet_to_hole_distance = 0.00156  # straight port throat before its R6.35 bend [m]
    He_port_overlap_inside_U = 0.0005  # port tongue overlap into the U void (fluid/fluid overlap) [m]
    He_port_mouth_overlap_middle_slot = 0.002  # port mouth overlap into the middle slot [m]
    He_port_mouth_overlap_rear_slot = 0.0005  # port mouth overlap into the rear slot [m]
    hole_to_PbLi_clearance = 0.0005  # He hole -> duct minimum distance in the 2D placement checks [m]
    hole_to_body_clearance = 0.0005  # He hole -> RAFM outline minimum distance in the 2D placement checks [m]
    He_network_boolean_mode = "united"  # 2-level unit cell, Z copies, 4 cuts (fast)
    Trim_He_networks_against_RAFM = False  # optional interface re-cut (removes ~0 mm^3)
    He_hole_boolean_batch_size = 16  # internal He holes subtracted from the RAFM 16 at a time

    MANIFOLD_SLOT_FRONT = 0  # index of the FRONT manifold slot (not connected)
    MANIFOLD_SLOT_MIDDLE = 1  # index of the MIDDLE manifold slot
    MANIFOLD_SLOT_REAR = 2  # index of the REAR manifold slot

    # ============================================================================
    # INPUTS -> inboard names (all values come from the USER INPUTS)
    # ============================================================================
    IB_FW_front_radius = R0 - a - tSOL  # plasma-facing first-wall radius [m] (3.500 m)
    He_outer_radial_thickness = 2.0 * He_channel_wall + He_channel_width  # wrapped-He band thickness [m] (38 mm)
    He_void_width = He_channel_width  # U void width [m]
    He_void_height = He_channel_height  # U void height [m]
    He_top_wall = 0.5 * He_channel_stack_wall  # RAFM above each level [m]
    He_bottom_wall = 0.5 * He_channel_stack_wall  # RAFM below each level [m]
    He_outer_bend_radius = FW_corner_radius  # FW corner radius [m]
    he_boundary_gap = 0.0  # the band's outer face is the sector wall
    He_U_fluid_end_fillet_radius = He_channel_end_fillet_radius  # U-leg end fillets [m]
    He_port_internal_fillet_radius = He_channel_end_fillet_radius  # port bends [m]
    He_manifold_port_radial_length = Manifold_port_width  # port mouth width [m]
    He_manifold_port_height = He_channel_height  # port height = channel height [m]
    He_manifold_slot_radial_length = Manifold_slot_length  # slot length [m]
    He_manifold_inter_slot_rib = Manifold_slot_gap  # RAFM between slots [m]
    He_manifold_rear_wall = Manifold_slot_gap  # RAFM behind the last slot [m]
    He_manifold_gap_from_wrapped_channel = Manifold_gap_to_wrapped_channel  # band -> slot outer side [m]
    He_manifold_inner_wall = Manifold_slot_to_housing_wall  # slot inner side -> housing wall [m]
    He_manifold_slot_corner_radius = Manifold_slot_fillet_radius  # slot corner radius (design) [m]
    Middle_manifold_slot_corner_radius = Manifold_slot_fillet_radius  # middle slot corner radius (design) [m]
    He_manifold_to_RAFM_inner_fillet_radius = Manifold_housing_fillet_radius  # housing <-> back wall fillet [m]
    num_rows = int(Num_PbLi_rows_IB)  # PbLi rows
    front_gap_after_He = 0.0  # PbLi row 1 starts at the band's inner face
    PbLi_edge_gap_from_wrapped_He = 0.0  # edge ducts start at the band's inner face
    PbLi_intercolumn_web_width = PbLi_column_gap  # web between duct columns [m]
    PbLi_corner_radius = PbLi_duct_fillet_radius  # duct corner radius [m]
    SiC_offset_from_PbLi_cavity = SiC_gap  # PbLi gap (sides, back) [m]
    SiC_wall_thickness = SiC_thickness  # SiC wall [m]
    SiC_front_gap_by_row = [SiC_front_gap_row1] + [SiC_gap] * (num_rows - 1)  # PbLi gap at each row's front [m]
    rear_He_hole_across_web = Rear_He_hole_length  # rear hole radial size [m]
    small_He_hole_toroidal_width = Column_He_hole_width  # column hole size across the web [m]

    # ============================================================================
    # END OF INBOARD SETTINGS
    # ============================================================================


    # corner radius currently used for each slot (the slot test may lower it)
    Slot_corner_radius_used = {MANIFOLD_SLOT_FRONT: He_manifold_slot_corner_radius,
                               MANIFOLD_SLOT_MIDDLE: Middle_manifold_slot_corner_radius,
                               MANIFOLD_SLOT_REAR: He_manifold_slot_corner_radius}

    def manifold_slot_corner_radius(slot_index):  # corner radius used for a given manifold slot
        return Slot_corner_radius_used[slot_index]  # set by the inputs / the slot test


    def port_mouth_overlap(slot_index):  # how far a port mouth overlaps into its slot
        if slot_index == MANIFOLD_SLOT_MIDDLE:  # middle-slot ports overlap more
            return He_port_mouth_overlap_middle_slot  # 2 mm into the middle slot
        return He_port_mouth_overlap_rear_slot  # 0.5 mm into the rear slot


    # ============================================================
    # DERIVED GLOBAL GEOMETRY
    # ============================================================

    Sector_angle = 360.0 / Num_sectors  # sector angle [deg] (22.5 for 16 sectors)

    sector_theta_min = 0.0  # sector starts at theta = 0
    sector_theta_max = Sector_angle  # sector ends at theta = Sector_angle

    if He_outer_radial_thickness <= He_void_width:  # the channel must be thicker than its void
        raise ValueError('He_outer_radial_thickness must be larger than He_void_width.')  # stop with a clear message

    He_side_wall = 0.5 * (He_outer_radial_thickness - He_void_width)  # steel wall on each side of the void [m] (4 mm)

    tube_width = He_outer_radial_thickness  # channel thickness used as the strip width

    if He_outer_bend_radius <= 0.5 * tube_width:  # bend radius must clear half the channel thickness
        # stop with a clear message
        raise ValueError('He_outer_bend_radius must exceed half of He_outer_radial_thickness.')

    He_centerline_bend_radius = He_outer_bend_radius - 0.5 * tube_width  # bend radius of the channel centerline [m]

    # The rounded-U centerline uses the derived centerline bend radius.
    bend_length = He_centerline_bend_radius

    tube_height = He_bottom_wall + He_void_height + He_top_wall  # one level = wall + void + wall [m] (20.12 mm)

    # Four levels: 2 + [16.12, 4, 16.12, 4, 16.12, 4, 16.12] + 2 = 80.48 mm
    # In the same slot: 24.12 mm between successive 16.12 mm openings
    tube_pitch = tube_height

    # ---------------------------------------------------------------------------
    # HEIGHT: matched to the outboard (same total height, centered on Z = 0).
    # As many full channel levels as fit; the leftover height is split equally
    # top and bottom as solid RAFM.
    # ---------------------------------------------------------------------------
    IB_height = 2.0 * Blanket_half_height
    Number_of_He_tubes = int(math.floor(IB_height / tube_pitch + 1.0e-9))  # full levels that fit in the IB height
    IB_extra_height = IB_height - Number_of_He_tubes * tube_pitch  # leftover height that cannot hold a full level [m]
    IB_z_bottom = -0.5 * IB_height  # bottom face of the IB slab (centered on Z = 0)
    IB_z_top = 0.5 * IB_height  # top face of the IB slab
    IB_first_level_z = IB_z_bottom + 0.5 * IB_extra_height  # half the leftover height goes below level 1 as solid RAFM
    # clear RAFM between two openings into the same slot [m]
    same_slot_clear_gap = tube_pitch * 2.0 - He_manifold_port_height
    model_height = IB_height                  # full slab height (with extra RAFM)

    # The front surface is fixed by IB_FW_front_radius.
    # Therefore the wrapped-He centerline is half a tube thickness inward.
    He_front_outer_radius = IB_FW_front_radius

    He_front_center_radius = IB_FW_front_radius - 0.5 * tube_width  # radius of the channel centerline at the FW [m]

    He_front_inner_radius = IB_FW_front_radius - tube_width  # inner (back) face of the FW channel band [m]

    tube_centerline_radius = He_front_center_radius  # radius used for the U centerline front arc

    # ------------------------------------------------------------
    # Derived radial stack, first wall -> back:
    #   He band | row 1 | row gap | row 2 | ... | last row | rear gap | rear He
    #   holes | gap | RAFM back wall | gap | 3 manifold slots (+ gaps)
    # ------------------------------------------------------------
    if num_rows < 1:  # need at least one row
        raise ValueError("Num_PbLi_rows_IB must be at least 1.")
    PbLi_row_bounds = []  # (front radius, back radius) of every PbLi row [m]
    _r = He_front_inner_radius - front_gap_after_He  # front of row 1
    for _i in range(num_rows):  # each row
        _L = PbLi_row1_length if _i == 0 else PbLi_other_rows_length  # its length
        PbLi_row_bounds.append((_r, _r - _L))  # store it
        _r -= _L + (PbLi_row_gap if _i < num_rows - 1 else 0.0)  # next row starts behind the row gap
    rear_He_row_front_radius = PbLi_row_bounds[-1][1] - Rear_He_gap_to_ducts  # front of the rear He holes [m]
    rear_He_row_back_radius = rear_He_row_front_radius - Rear_He_hole_length  # back of the rear He holes [m]
    IB_body_back_radius = rear_He_row_back_radius - Rear_He_gap_to_back_wall  # RAFM back wall [m]
    rear_structural_thickness = Rear_He_gap_to_ducts + Rear_He_hole_length + Rear_He_gap_to_back_wall  # [m]
    total_RAFM_depth = IB_FW_front_radius - IB_body_back_radius  # FW -> back wall [m]

    # ------------------------------------------------------------
    # Derived PbLi side geometry used by the manifold
    # ------------------------------------------------------------

    # offset of ducts 1/5 from the sector walls (He band + gap) [m]
    PbLi_side_edge_offset_from_sector_boundary = tube_width + PbLi_edge_gap_from_wrapped_He

    # ------------------------------------------------------------
    # Derived manifold front/back locations
    # ------------------------------------------------------------

    # The first slot FRONT edge is a circular arc concentric with the machine,
    # Manifold_back_wall_to_first_slot behind the RAFM back wall.
    He_manifold_slot1_front_axis_coordinate = IB_body_back_radius - Manifold_back_wall_to_first_slot

    # This is a useful check only; with the reference geometry it is exactly 4 mm.
    He_manifold_front_slot_offset_from_RAFM_back = IB_body_back_radius - He_manifold_slot1_front_axis_coordinate

    # radial span of the 3 slots + 2 ribs [m]
    He_manifold_used_slot_span = 3.0 * He_manifold_slot_radial_length + 2.0 * He_manifold_inter_slot_rib

    # Manifold REAR face is also a concentric circular arc.
    He_manifold_back_axis_coordinate = (
        He_manifold_slot1_front_axis_coordinate
        - He_manifold_used_slot_span
        - He_manifold_rear_wall
    )

    # radial depth of the manifold zone behind the main body [m]
    He_manifold_total_radial_depth = IB_body_back_radius - He_manifold_back_axis_coordinate

    # Manifold housing INNER walls: set from the last duct row's layout (see the duct block).

    # Main RAFM back edge is now an exact circle at this radius.
    He_main_RAFM_back_edge_radius = IB_body_back_radius

    # offset of the U side-leg centerline from the sector wall [m]
    _wrapped_leg_center_offset = 0.5 * tube_width + he_boundary_gap

    # Straight sector-side outer edge terminates on manifold rear radius.
    # This is a single full-thickness RAFM contour, not separate tube ends.
    He_side_leg_back_radius = He_manifold_back_axis_coordinate

    # length of each straight U side leg [m]
    straight_side_leg_length = tube_centerline_radius - He_side_leg_back_radius - bend_length

    # Side-leg centerline is half a tube width inward from the sector wall.
    he_side_centerline_offset = 0.5 * tube_width + he_boundary_gap

    # Physical INNER edge of wrapped-He FLUID on a straight side leg.
    He_side_fluid_inner_offset_from_sector_wall = 0.5 * tube_width + 0.5 * He_void_width

    # Physical INNER SOLID edge of the wrapped-He channel.
    He_wrapped_channel_inner_solid_offset_from_sector_wall = tube_width

    # OUTER edge of every manifold slot, separated from the wrapped-He solid by
    # the user-defined manufacturing ligament.
    He_manifold_slot_outer_offset_from_sector_wall = (
        He_wrapped_channel_inner_solid_offset_from_sector_wall
        + He_manifold_gap_from_wrapped_channel
    )

    # Total solid thickness bridged by the connection port:
    # wrapped-He side wall + manufacturing ligament.
    He_connection_wall_thickness = (
        He_manifold_slot_outer_offset_from_sector_wall
        - He_side_fluid_inner_offset_from_sector_wall
    )

    # Exact side-leg coordinate required so the INNER RAFM back edge lands on the
    # requested circular back radius rather than slightly outside it.
    IB_body_back_side_leg_coordinate = math.sqrt(max(0.0, IB_body_back_radius ** 2 - tube_width ** 2))


    # ------------------------------------------------------------
    # Derived SiC insert geometry
    # ------------------------------------------------------------

    # total inset from the cavity wall to the PbLi core [m] (7 mm)
    SiC_total_inset_to_inner_fluid = SiC_offset_from_PbLi_cavity + SiC_wall_thickness

    SiC_outer_corner_radius = PbLi_corner_radius  # SiC outer surface: same corner radius as the duct [m]
    SiC_inner_corner_radius_center = PbLi_corner_radius  # SiC inner surface, inner ducts: same as the duct [m]
    SiC_inner_corner_radius_edge = PbLi_corner_radius - SiC_wall_thickness  # SiC inner surface, edge ducts [m]
    SiC_inner_corner_radius = SiC_inner_corner_radius_edge  # smallest SiC corner radius [m]

    if SiC_offset_from_PbLi_cavity <= 0.0:  # validate: PbLi gap must be positive
        raise ValueError('SiC_offset_from_PbLi_cavity must be > 0.')  # stop with a clear message

    if SiC_wall_thickness <= 0.0:  # validate: SiC wall must be positive
        raise ValueError('SiC_wall_thickness must be > 0.')  # stop with a clear message

    if SiC_outer_corner_radius <= 0.0:  # validate: SiC outer corner radius must stay positive
        raise ValueError('SiC_offset_from_PbLi_cavity is too large for PbLi_corner_radius.')  # stop with a clear message

    if SiC_inner_corner_radius <= 0.0:  # validate: SiC inner corner radius must stay positive
        raise ValueError('SiC offset + wall thickness is too large for PbLi_corner_radius.')  # stop with a clear message


    # ============================================================
    # DISCRETIZATION / CUT DEPTH
    # ============================================================

    Num_side_leg_points = 2       # straight legs: 2 points = 1 flat face
    Num_bend_points = 12          # 7.5 deg facets (<0.2 mm chord error)
    Num_front_arc_points = 30     # ~75 um chord error on R 3.5 m
    Num_body_arc_points = 90

    Num_hole_corner_points = 4  # polyline points per small He-hole corner

    void_end_extension = 0.020  # extra length for void cutters past their ends [m]

    cut_z_min = IB_z_bottom  # lowest Z any cutter must reach
    cut_z_max = IB_z_top  # highest Z any cutter must reach
    cut_z_eps = 0.002  # small Z overshoot for cutters [m]


    # ============================================================
    # BASIC XY HELPERS
    # ============================================================

    def point_xy(radius, theta_deg):  # XY point at (radius, angle)
        theta = math.radians(theta_deg)  # degrees -> radians
        return [radius * math.cos(theta), radius * math.sin(theta)]  # polar -> Cartesian


    def radial_unit(theta_deg):  # unit vector pointing radially outward at theta
        theta = math.radians(theta_deg)  # degrees -> radians
        return [math.cos(theta), math.sin(theta)]  # (cos, sin)


    def tangent_unit(theta_deg):  # unit vector along +theta (counter-clockwise)
        theta = math.radians(theta_deg)  # degrees -> radians
        return [-math.sin(theta), math.cos(theta)]  # (-sin, cos)


    def add_xy(p, q):  # 2D vector sum
        return [p[0] + q[0], p[1] + q[1]]  # p + q


    def sub_xy(p, q):  # 2D vector difference
        return [p[0] - q[0], p[1] - q[1]]  # p - q


    def scale_xy(p, scale):  # scale a 2D vector
        return [p[0] * scale, p[1] * scale]  # p * scale


    def length_xy(p):  # length of a 2D vector
        return math.sqrt(p[0] * p[0] + p[1] * p[1])  # sqrt(x^2 + y^2)


    def normalize_xy(p):  # unit vector in the direction of p
        magnitude = length_xy(p)  # vector length

        if magnitude <= 0.0:  # a zero vector has no direction
            raise ValueError('Cannot normalize a zero-length vector.')  # stop with a clear message

        return [p[0] / magnitude, p[1] / magnitude]  # p / |p|


    # ============================================================
    # CUBIT HELPERS
    # ============================================================

    def create_vertex_xyz(x, y, z):  # create a Cubit vertex at (x, y, z) and return its ID
        cubit.cmd(f'create vertex {x} {y} {z}')  # Cubit command

        return cubit.get_last_id('vertex')  # ID of the vertex just made


    def points_are_close(p1, p2, tol=1.0e-9):  # True if two XY points coincide within tol
        return length_xy(sub_xy(p1, p2)) < tol  # distance check


    def clean_polygon_points(points_xy, tol=1.0e-9):  # drop duplicate consecutive points (and a repeated closing point)
        cleaned = []  # output list

        for point in points_xy:  # walk the input points
            if not cleaned:  # first point is always kept
                cleaned.append(point)  # keep it

            elif not points_are_close(point, cleaned[-1], tol):  # keep only points that differ from the previous one
                cleaned.append(point)  # keep it

        # polygon given closed (last = first): drop the duplicate
        if len(cleaned) > 1 and points_are_close(cleaned[0], cleaned[-1], tol):
            cleaned.pop()  # remove the closing point

        if len(cleaned) < 3:  # a face needs at least 3 distinct corners
            raise ValueError('Polygon has fewer than 3 unique points.')  # stop with a clear message

        return cleaned  # cleaned polygon


    def get_largest_volume_id(volume_ids):  # ID of the largest volume in a list (by volume)
        if not volume_ids:  # empty list
            return None  # nothing to choose

        largest_id = volume_ids[0]  # start with the first
        largest_volume = cubit.get_total_volume([largest_id])  # its volume

        for volume_id in volume_ids[1:]:  # compare with the others
            volume = cubit.get_total_volume([volume_id])  # this volume's size

            if volume > largest_volume:  # bigger one found
                largest_id = volume_id  # remember its ID
                largest_volume = volume  # and its size

        return largest_id  # largest volume ID


    def make_polygon_surface_vertex(points_xy, z_value):  # planar surface through polygon vertices at height z_value
        points_xy = clean_polygon_points(points_xy)  # remove duplicate points

        # one Cubit vertex per corner
        vertex_ids = [create_vertex_xyz(point[0], point[1], z_value) for point in points_xy]

        before = set(cubit.get_entities('surface'))  # surfaces before the command

        # Cubit: surface through the ordered vertices
        cubit.cmd('create surface vertex ' + ' '.join(map(str, vertex_ids)))

        after = set(cubit.get_entities('surface'))  # surfaces after the command

        new_surfaces = list(after - before)  # the new surface(s)

        if new_surfaces:  # normally exactly one
            return max(new_surfaces)  # newest surface ID

        return cubit.get_last_id('surface')  # fallback: last surface ID


    # planar surface from straight curves joining polygon corners
    def make_polygon_surface_curve_loop(points_xy, z_value):
        points_xy = clean_polygon_points(points_xy)  # remove duplicate points

        # one Cubit vertex per corner
        vertex_ids = [create_vertex_xyz(point[0], point[1], z_value) for point in points_xy]

        curve_ids = []  # curve IDs around the loop

        for i in range(len(vertex_ids)):  # each edge of the polygon
            v1 = vertex_ids[i]  # edge start vertex
            v2 = vertex_ids[(i + 1) % len(vertex_ids)]  # edge end vertex (wraps to the first)

            cubit.cmd(f'create curve vertex {v1} {v2}')  # straight curve between them

            curve_ids.append(cubit.get_last_id('curve'))  # remember the curve

        before = set(cubit.get_entities('surface'))  # surfaces before the command

        # Cubit: surface bounded by the closed curve loop
        cubit.cmd('create surface curve ' + ' '.join(map(str, curve_ids)))

        after = set(cubit.get_entities('surface'))  # surfaces after the command

        new_surfaces = list(after - before)  # the new surface(s)

        if not new_surfaces:  # nothing was made
            raise RuntimeError('Surface creation from curve loop failed.')  # stop with a clear message

        return max(new_surfaces)  # newest surface ID


    def sweep_surface_in_z(surface_id, height):  # extrude a surface straight up in +Z by height; return the volume ID
        before = set(cubit.get_entities('volume'))  # volumes before the sweep

        try:  # try the standard syntax first
            cubit.cmd(f'sweep surface {surface_id} direction 0 0 1 distance {height}')  # Cubit: extrude along +Z

        except Exception:
            cubit.cmd(f'sweep surface {surface_id} vector 0 0 {height}')  # alternative syntax for other Cubit versions

        after = set(cubit.get_entities('volume'))  # volumes after the sweep

        new_volumes = list(after - before)  # volumes the sweep created

        if new_volumes:  # normally one
            return get_largest_volume_id(new_volumes)  # keep the largest (the solid)

        return cubit.get_last_id('volume')  # fallback: last volume ID


    # polygon (vertex-built face) extruded from z_min to z_max
    def make_prism_from_polygon_vertex(points_xy, z_min, z_max):
        surface_id = make_polygon_surface_vertex(points_xy, z_min)  # bottom face

        return sweep_surface_in_z(surface_id, z_max - z_min)  # extrude by the height


    # polygon (curve-loop face) extruded from z_min to z_max
    def make_prism_from_polygon_curve_loop(points_xy, z_min, z_max):
        surface_id = make_polygon_surface_curve_loop(points_xy, z_min)  # bottom face

        return sweep_surface_in_z(surface_id, z_max - z_min)  # extrude by the height


    def resolve_result(original_volume_id, before_volumes):  # ID of the body after a Boolean (Cubit may renumber it)
        after = set(cubit.get_entities('volume'))  # volumes now present

        if original_volume_id in after:  # ID kept: nothing to resolve
            return original_volume_id  # same ID

        new_volumes = list(after - set(before_volumes))  # IDs that appeared during the Boolean

        if new_volumes:  # the body was given a new ID
            return get_largest_volume_id(new_volumes)  # largest new volume = the body

        if after:  # odd case: fall back to the largest volume overall
            return get_largest_volume_id(list(after))  # largest remaining volume

        raise RuntimeError('No volume exists after Boolean operation.')  # nothing left at all: stop


    # ============================================================
    # VOLUME IDENTIFICATION
    # ============================================================

    def register_volume(volume_id, volume_name, family_name=None):  # give a volume a readable name in Cubit
        # Keep every body individually identifiable in the Cubit model.
        try:
            cubit.cmd(f"volume {volume_id} name '{volume_name}'")  # Cubit: name the volume
        except Exception:
            pass  # naming is cosmetic; ignore failures


    # ============================================================
    # FAST PATTERN / BOOLEAN HELPERS
    # ============================================================

    def copy_volume_move_xy(volume_id, dx, dy):  # translate-copy a volume in XY; return the new ID
        before = set(cubit.get_entities('volume'))  # volumes before the copy

        # equivalent command spellings for different Cubit versions
        commands = [
            (
                f"volume {volume_id} copy "
                f"move {dx} {dy} 0"
            ),
            (
                f"copy volume {volume_id} "
                f"move {dx} {dy} 0"
            ),
            (
                f"volume {volume_id} copy "
                f"move x {dx} y {dy} z 0"
            ),
            (
                f"copy volume {volume_id} "
                f"move x {dx} y {dy} z 0"
            ),
        ]

        for command in commands:  # try each spelling until one works
            try:  # a failed spelling raises an error
                cubit.cmd(command)  # Cubit: copy + move

                after = set(cubit.get_entities('volume'))  # volumes after the command

                new_volumes = list(after - before)  # IDs created

                if new_volumes:  # the copy exists
                    return get_largest_volume_id(new_volumes)  # return its ID

            except Exception:
                pass  # try the next spelling

        raise RuntimeError('Could not copy/move He-hole prototype.')  # no spelling worked: stop


    def copy_volume_rotate_z(volume_id, angle_deg):  # rotate-copy a volume about the Z axis; return the new ID
        before = set(cubit.get_entities('volume'))  # volumes before the copy

        # equivalent command spellings for different Cubit versions
        commands = [
            (
                f"volume {volume_id} copy "
                f"rotate {angle_deg} about z"
            ),
            (
                f"copy volume {volume_id} "
                f"rotate {angle_deg} about z"
            ),
            (
                f"volume {volume_id} copy "
                f"rotate angle {angle_deg} about z"
            ),
        ]

        for command in commands:  # try each spelling until one works
            try:  # a failed spelling raises an error
                cubit.cmd(command)  # Cubit: copy + rotate

                after = set(cubit.get_entities('volume'))  # volumes after the command

                new_volumes = list(after - before)  # IDs created

                if new_volumes:  # the copy exists
                    return get_largest_volume_id(new_volumes)  # return its ID

            except Exception:
                pass  # try the next spelling

        raise RuntimeError('Could not rotate-copy He-hole prototype.')  # no spelling worked: stop


    # subtract cutters from a body in batches, keeping the cutters (fluids)
    def subtract_cutter_batch_from_body(body_volume_id, cutter_volume_ids, label, batch_size, progress_state=None):
        if not cutter_volume_ids:  # nothing to cut
            return body_volume_id  # body unchanged

        total_batches = int(math.ceil(len(cutter_volume_ids) / float(batch_size)))  # number of batches

        for batch_index in range(total_batches):  # one Boolean per batch
            start = batch_index * batch_size  # first cutter index of this batch

            end = min(len(cutter_volume_ids), start + batch_size)  # one past the last cutter index

            batch = cutter_volume_ids[start:end]  # cutter IDs in this batch

            # progress message
            print(f'  {label}: Boolean batch {batch_index + 1}/{total_batches} ({len(batch)} cutters)')

            before = set(cubit.get_entities('volume'))  # volumes before the Boolean

            # Cubit: subtract the batch, imprint, keep the cutters
            cubit.cmd(
                "subtract volume "
                + " ".join(
                    map(
                        str,
                        batch,
                    )
                )
                + f" from volume "
                f"{body_volume_id} imprint keep_tool"
            )

            body_volume_id = resolve_result(body_volume_id, before)  # the body's (possibly new) ID

            if progress_state is not None:  # optional running total for the progress message
                progress_state['completed'] += len(batch)  # cutters done so far

                # progress message
                print(f"  IB internal He-hole CUT progress: {progress_state['completed']}/{progress_state['total']}")

        return body_volume_id  # final body ID


    # ============================================================
    # PATH HELPERS
    # ============================================================


    # ============================================================
    # WRAPPED He CENTERLINE
    # ============================================================



    # ============================================================
    # WRAPPED He SOLIDS + ALTERNATING MANIFOLD PORTS
    # ============================================================


    # ============================================================
    # BLANKET BODY
    # ============================================================



    def make_blanket_body_volume(body_polygon):  # extrude the body outline over the full IB height
        return make_prism_from_polygon_vertex(body_polygon, IB_z_bottom, IB_z_top)  # prism from IB_z_bottom to IB_z_top


    # ============================================================
    # BACK MANIFOLD GEOMETRY
    # ============================================================



    # directions and normals of one side's manifold (left or right)
    def get_manifold_side_geometry(
        side_name,
    ):
        # OUTER manifold / slot edge:
        #   parallel to the straight wrapped-He side channel
        #
        # INNER manifold wall:
        #   exactly collinear with the outer wall of PbLi duct 2 (left)
        #   or PbLi duct 4 (right)
        #
        # These two directions are intentionally slightly different.
        if side_name == "left":
            boundary_theta = sector_theta_min  # left sector wall at theta = 0

            boundary_inward_sign = +1.0  # interior is toward +theta

            inner_wall_theta = He_manifold_left_inner_wall_theta  # left manifold inner wall angle

            # Points from the PbLi-collinear inner wall toward the wrapped-He side.
            inner_wall_outward_normal = scale_xy(tangent_unit(inner_wall_theta), -1.0)

        elif side_name == "right":  # right side
            boundary_theta = sector_theta_max  # right sector wall at theta = max

            boundary_inward_sign = -1.0  # interior is toward -theta

            inner_wall_theta = He_manifold_right_inner_wall_theta  # right manifold inner wall angle

            # points from the inner wall toward the wrapped-He side
            inner_wall_outward_normal = tangent_unit(inner_wall_theta)

        else:
            raise ValueError("side_name must be 'left' or 'right'.")  # stop with a clear message

        boundary_direction = radial_unit(boundary_theta)  # direction along the sector wall

        # normal from the wall into the blanket
        boundary_inward_normal = scale_xy(tangent_unit(boundary_theta), boundary_inward_sign)

        inner_wall_direction = radial_unit(inner_wall_theta)  # direction along the manifold inner wall

        # everything the slot / port builders need
        return {
            "side_name":
                side_name,

            "boundary_theta":
                boundary_theta,

            # Direction of wrapped-He side / slot OUTER long edge.
            "outer_line_direction":
                boundary_direction,

            # Normal pointing from the wrapped-He side toward the blanket interior.
            "outer_line_inward_normal":
                boundary_inward_normal,

            # Direction of manifold inner wall / slot INNER long edge.
            "inner_line_direction":
                inner_wall_direction,

            # Normal pointing from manifold inner wall toward the slots.
            "inner_line_outward_normal":
                inner_wall_outward_normal,

            # Housing wall = the radial line above moved by this amount along that
            # normal (negative = half a web toward the centre: collinear with the duct side).
            "inner_wall_offset":
                He_manifold_inner_wall_line_offset,
        }


    # point where an offset straight line meets the circle |p| = radius
    def point_on_offset_line_at_radius(line_direction, line_normal, line_offset, radius):
        # A straight offset line can be written
        #
        #   p = s * line_direction + line_offset * line_normal
        #
        # with line_direction and line_normal orthonormal.
        #
        # Intersect it with the circle |p| = radius.  We want the positive-radial
        # intersection used by this inboard blanket geometry.
        radial_part_sq = radius ** 2 - line_offset ** 2

        if radial_part_sq <= 0.0:  # the line must reach that radius
            # stop with a clear message
            raise ValueError('Requested manifold line does not intersect the requested radius.')

        radial_part = math.sqrt(radial_part_sq)  # radial component of the intersection

        return add_xy(scale_xy(line_direction, radial_part), scale_xy(line_normal, line_offset))  # intersection point


    # fillet circle between a straight slot edge and a concentric slot arc
    def line_circle_fillet_data(
        line_direction,
        interior_normal,
        boundary_offset,
        boundary_radius,
        fillet_radius,
        circle_is_front,
    ):
        # Construct a constant-radius fillet between:
        #   1) one straight slot long edge
        #   2) one concentric front/back slot arc
        #
        # interior_normal points INTO the slot from the straight boundary.
        #
        # FRONT circle:
        #   slot is radially inward of the circle -> center radius R - r
        #
        # BACK circle:
        #   slot is radially outward of the circle -> center radius R + r
        if circle_is_front:
            center_radius = boundary_radius - fillet_radius  # fillet center radius (inside the front arc)
        else:
            center_radius = boundary_radius + fillet_radius  # fillet center radius (outside the back arc)

        center_line_offset = boundary_offset + fillet_radius  # fillet center offset from the straight edge

        # fillet center
        center = point_on_offset_line_at_radius(line_direction, interior_normal, center_line_offset, center_radius)

        line_tangent = add_xy(center, scale_xy(interior_normal, -fillet_radius))  # tangent point on the straight edge

        # tangent point on the arc (same polar angle as the center)
        circle_tangent = scale_xy(center, boundary_radius / center_radius)

        # fillet center + both tangent points
        return {'center': center, 'line_tangent': line_tangent, 'circle_tangent': circle_tangent}


    def manifold_slot_front_radius(slot_index):  # radius of a slot's FRONT arc
        if slot_index not in (MANIFOLD_SLOT_FRONT, MANIFOLD_SLOT_MIDDLE, MANIFOLD_SLOT_REAR):  # only slots 0, 1, 2 exist
            raise ValueError('Invalid manifold slot index.')  # stop with a clear message

        # front of slot 1 minus (slot + rib) per slot index
        return (
            He_manifold_slot1_front_axis_coordinate
            - slot_index
            * (
                He_manifold_slot_radial_length
                + He_manifold_inter_slot_rib
            )
        )


    def manifold_slot_back_radius(slot_index):  # radius of a slot's BACK arc
        # front radius minus the slot length
        return manifold_slot_front_radius(slot_index) - He_manifold_slot_radial_length


    def manifold_slot_center_radius(slot_index):  # radius of a slot's mid line
        # average of front and back
        return 0.5 * (manifold_slot_front_radius(slot_index) + manifold_slot_back_radius(slot_index))


    def get_slot_line_data(side_name):  # offset lines bounding the slots on one side
        geometry = get_manifold_side_geometry(side_name)  # side directions / normals

        # OUTER slot edge:
        #   parallel to wrapped-He side channel
        #   wrapped-He INNER SOLID edge + 4 mm RAFM manufacturing ligament
        outer_direction = geometry['outer_line_direction']

        outer_interior_normal = geometry['outer_line_inward_normal']  # normal into the slot from the outer edge

        # outer edge offset from the sector wall [m] (42 mm)
        outer_offset = He_manifold_slot_outer_offset_from_sector_wall

        # INNER slot edge:
        #   parallel to / offset from the PbLi-collinear manifold inner wall
        #   by exactly 10.16 mm toward the slot.
        inner_direction = geometry['inner_line_direction']

        inner_interior_normal = geometry['inner_line_outward_normal']  # normal into the slot from the inner edge

        inner_offset = geometry["inner_wall_offset"] + He_manifold_inner_wall  # 10.16 mm from the housing wall [m]

        # line data for the slot builders
        return {
            "geometry":
                geometry,

            "outer_direction":
                outer_direction,

            "outer_interior_normal":
                outer_interior_normal,

            "outer_offset":
                outer_offset,

            "inner_direction":
                inner_direction,

            "inner_interior_normal":
                inner_interior_normal,

            "inner_offset":
                inner_offset,
        }


    # ============================================================
    # ROUNDED RECTANGLE HELPERS
    # ============================================================


    # ============================================================
    # PbLi LAYOUT
    # ============================================================


    # cut the 10 PbLi ducts from the IB RAFM, keeping each cutter as PbLi fluid
    def cut_PbLi_ducts(blanket_body_id, layout):
        print("")  # blank line in the console
        print('Cutting PbLi ducts and KEEPING the cutters as PbLi fluid bodies...')  # stage message

        fluid_ids = []  # retained PbLi bodies

        for (i, duct) in enumerate(layout, start=1):  # each duct
            # Exact physical height because this tool survives as the PbLi body.
            cutter_id = make_prism_from_polygon_curve_loop(duct['polygon'], cut_z_min, cut_z_max)

            before = set(cubit.get_entities('volume'))  # volumes before the Boolean

            # Cubit: cut the duct, imprint, keep the cutter
            cubit.cmd(f'subtract volume {cutter_id} from volume {blanket_body_id} imprint keep_tool')

            blanket_body_id = resolve_result(blanket_body_id, before)  # the RAFM's (possibly new) ID

            fluid_ids.append(cutter_id)  # keep the cutter as a PbLi body

            # name it IB_PbLi_fluid_Rr_Cc
            register_volume(
                cutter_id,
                (
                    f"IB_PbLi_fluid_"
                    f"R{duct['row_index'] + 1}_"
                    f"C{duct['col_index'] + 1}"
                ),
                "IB_PbLi_fluid",
            )

            print(f'  PbLi cut/fluid body {i}/{len(layout)}')  # progress message

        return (blanket_body_id, fluid_ids)  # updated RAFM ID + PbLi IDs


    # ============================================================
    # SiC INSERTS INSIDE PbLi DUCTS
    # ============================================================



    # split each IB PbLi duct into PbLi gap / SiC shell / PbLi core
    def build_SiC_inserts_and_split_PbLi(pbli_layout, original_pbli_fluid_ids):
        if len(pbli_layout) != len(original_pbli_fluid_ids):  # one PbLi body per duct expected
            # stop with a clear message
            raise RuntimeError('PbLi layout / retained-fluid count mismatch before SiC build.')

        final_pbli_fluid_ids = []  # output PbLi bodies (gap + core per duct)
        sic_solid_ids = []  # output SiC shells

        print("")  # blank line in the console
        print('Building hollow SiC inserts inside all PbLi ducts...')  # stage message

        # each duct and its PbLi body
        for (i, (duct, original_pbli_id)) in enumerate(zip(pbli_layout, original_pbli_fluid_ids), start=1):
            row_label = duct['row_index'] + 1  # row number for names

            col_label = duct['col_index'] + 1  # column number for names

            # ----------------------------------------------------
            # OUTER SiC envelope: 2 mm inward from PbLi cavity edge
            # ----------------------------------------------------

            # SiC outer surface prism (2 mm inset)
            sic_outer_envelope_id = (
                make_prism_from_polygon_curve_loop(
                    make_inset_PbLi_polygon(
                        duct,
                        SiC_offset_from_PbLi_cavity,
                    ),
                    cut_z_min,
                    cut_z_max,
                )
            )

            # Remove the envelope from the original PbLi body while preserving it.
            # The target becomes the PbLi fluid GAP outside the SiC insert.
            before = set(cubit.get_entities('volume'))

            # Cubit: cut it from the PbLi body, keep it -> PbLi gap remains
            cubit.cmd(f'subtract volume {sic_outer_envelope_id} from volume {original_pbli_id} imprint keep_tool')

            pbli_outer_gap_id = resolve_result(original_pbli_id, before)  # PbLi gap ID

            # ----------------------------------------------------
            # INNER PbLi core: another 5 mm inward
            # ----------------------------------------------------

            # PbLi core prism (7 mm inset)
            pbli_inner_core_id = (
                make_prism_from_polygon_curve_loop(
                    make_inset_PbLi_polygon(
                        duct,
                        SiC_total_inset_to_inner_fluid,
                    ),
                    cut_z_min,
                    cut_z_max,
                )
            )

            # Hollow the preserved SiC envelope. KEEP_TOOL leaves the inner core
            # itself as the physical PbLi fluid inside the SiC insert.
            before = set(cubit.get_entities('volume'))

            # Cubit: cut the core from the SiC envelope, keep the core -> SiC shell
            cubit.cmd(f'subtract volume {pbli_inner_core_id} from volume {sic_outer_envelope_id} imprint keep_tool')

            sic_shell_id = resolve_result(sic_outer_envelope_id, before)  # SiC shell ID

            # name the PbLi gap
            register_volume(pbli_outer_gap_id, f'IB_PbLi_outer_gap_R{row_label}_C{col_label}', 'IB_PbLi_fluid')

            # name the PbLi core
            register_volume(pbli_inner_core_id, f'IB_PbLi_inner_core_R{row_label}_C{col_label}', 'IB_PbLi_fluid')

            # name the SiC shell
            register_volume(sic_shell_id, f'IB_SiC_insert_R{row_label}_C{col_label}', 'IB_SiC_solid')

            final_pbli_fluid_ids.extend([pbli_outer_gap_id, pbli_inner_core_id])  # record both PbLi bodies

            sic_solid_ids.append(sic_shell_id)  # record the SiC shell

            print(f'  SiC insert {i}/{len(pbli_layout)} (R{row_label} C{col_label})')  # progress message

        return (final_pbli_fluid_ids, sic_solid_ids)  # PbLi IDs + SiC IDs


    # ============================================================
    # 2D SAFETY HELPERS
    # ============================================================

    def polygon_centroid(polygon):  # average of the polygon points
        # (mean x, mean y)
        return [
            sum(
                point[0]
                for point in polygon
            )
            / len(polygon),

            sum(
                point[1]
                for point in polygon
            )
            / len(polygon),
        ]


    def point_in_polygon(point, polygon):  # ray-casting point-in-polygon test
        x = point[0]  # test point x
        y = point[1]  # test point y

        inside = False  # crossing parity

        j = len(polygon) - 1  # previous vertex index

        for i in range(len(polygon)):  # each edge (j -> i)
            xi = polygon[i][0]  # edge end x
            yi = polygon[i][1]  # edge end y

            xj = polygon[j][0]  # edge start x
            yj = polygon[j][1]  # edge start y

            if (yi > y) != (yj > y):  # edge straddles the horizontal line through the point
                denominator = yj - yi  # edge height

                if abs(denominator) < 1e-15:  # avoid division by zero
                    denominator = 1e-15  # tiny non-zero value

                crossing_x = (xj - xi) * (y - yi) / denominator + xi  # x where the edge crosses that line

                if x < crossing_x:  # crossing lies to the right of the point
                    inside = not inside  # flip the parity

            j = i  # advance

        return inside  # True if inside


    def cross2(a, b, c):  # 2D cross product (b - a) x (c - a)
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])  # signed area x 2


    def on_segment(a, b, point, tol=1.0e-10):  # True if point lies on segment a-b (within tol)
        if abs(cross2(a, b, point)) > tol:  # not collinear
            return False  # so not on the segment

        # inside the segment's bounding box
        return (
            min(
                a[0],
                b[0],
            )
            - tol
            <= point[0]
            <= max(
                a[0],
                b[0],
            )
            + tol

            and

            min(
                a[1],
                b[1],
            )
            - tol
            <= point[1]
            <= max(
                a[1],
                b[1],
            )
            + tol
        )


    def segments_intersect(a, b, c, d):  # True if segments a-b and c-d intersect or touch
        c1 = cross2(a, b, c)  # side of c relative to a-b

        c2 = cross2(a, b, d)  # side of d relative to a-b

        c3 = cross2(c, d, a)  # side of a relative to c-d

        c4 = cross2(c, d, b)  # side of b relative to c-d

        # proper crossing: each segment straddles the other
        if (c1 > 0.0 and c2 < 0.0 or (c1 < 0.0 and c2 > 0.0)) and (c3 > 0.0 and c4 < 0.0 or (c3 < 0.0 and c4 > 0.0)):
            return True  # they cross

        # touching cases (an endpoint on the other segment)
        return on_segment(a, b, c) or on_segment(a, b, d) or on_segment(c, d, a) or on_segment(c, d, b)


    def point_segment_distance(point, a, b):  # shortest distance from a point to segment a-b
        ab = sub_xy(b, a)  # segment vector

        ap = sub_xy(point, a)  # vector from a to the point

        denominator = ab[0] * ab[0] + ab[1] * ab[1]  # segment length squared

        if denominator <= 1.0e-20:  # degenerate segment
            return length_xy(sub_xy(point, a))  # distance to a

        t = (ap[0] * ab[0] + ap[1] * ab[1]) / denominator  # projection parameter along the segment

        t = max(0.0, min(1.0, t))  # clamp onto the segment

        closest = [a[0] + t * ab[0], a[1] + t * ab[1]]  # closest point on the segment

        return length_xy(sub_xy(point, closest))  # distance to it


    def polygon_edge_distance(polygon_a, polygon_b):  # minimum distance between two polygon outlines (0 if they cross)
        minimum = float('inf')  # start at infinity

        for i in range(len(polygon_a)):  # each edge of polygon A
            a0 = polygon_a[i]  # edge start

            a1 = polygon_a[(i + 1) % len(polygon_a)]  # edge end

            for j in range(len(polygon_b)):  # each edge of polygon B
                b0 = polygon_b[j]  # edge start

                b1 = polygon_b[(j + 1) % len(polygon_b)]  # edge end

                if segments_intersect(a0, a1, b0, b1):  # crossing edges
                    return 0.0  # outlines touch

                # smallest endpoint-to-edge distance so far
                minimum = min(
                    minimum,

                    point_segment_distance(
                        a0,
                        b0,
                        b1,
                    ),

                    point_segment_distance(
                        a1,
                        b0,
                        b1,
                    ),

                    point_segment_distance(
                        b0,
                        a0,
                        a1,
                    ),

                    point_segment_distance(
                        b1,
                        a0,
                        a1,
                    ),
                )

        return minimum  # minimum outline distance


    def polygons_overlap(polygon_a, polygon_b):  # True if two polygons cross or one lies inside the other
        if polygon_edge_distance(polygon_a, polygon_b) <= 1e-10:  # outlines touch or cross
            return True  # overlap

        if point_in_polygon(polygon_a[0], polygon_b):  # A inside B
            return True  # overlap

        if point_in_polygon(polygon_b[0], polygon_a):  # B inside A
            return True  # overlap

        return False  # separate


    # True if a candidate He hole fits inside the RAFM with clearance
    def candidate_hole_is_safe(candidate_polygon, body_polygon, pbli_layout):
        center = polygon_centroid(candidate_polygon)  # candidate center

        if not point_in_polygon(center, body_polygon):  # center must be inside the body outline
            return False  # reject

        for point in candidate_polygon:  # every candidate corner
            if not point_in_polygon(point, body_polygon):  # a corner outside the body
                return False  # reject

        # too close to the body outline
        if polygon_edge_distance(candidate_polygon, body_polygon) < hole_to_body_clearance:
            return False  # reject

        for duct in pbli_layout:  # check against every PbLi duct
            opening = duct['polygon']  # duct outline

            if polygons_overlap(candidate_polygon, opening):  # hole overlaps the duct
                return False  # reject

            if polygon_edge_distance(candidate_polygon, opening) < hole_to_PbLi_clearance:  # hole too close to the duct
                return False  # reject

        return True  # passes every check


    # ============================================================
    # He-HOLE PATTERN CALCULATORS
    # ============================================================


    # ============================================================
    # SMALL-WEB GEOMETRY
    # ============================================================


    # ============================================================
    # PRIMARY GEOMETRY / INPUT VALIDATION
    # ============================================================



    # ============================================================
    # COMPLETE He-HOLE PATTERN
    # ============================================================



    # ============================================================================
    # DRIVER-LEVEL HELPERS
    # ============================================================================

    def copy_volume_move(volume_id, dx, dy, dz):  # copy a volume and translate it by (dx, dy, dz)
        """Copy a volume and translate the copy; return the new volume ID."""
        before = set(cubit.get_entities("volume"))
        # equivalent command spellings
        commands = [
            f"volume {volume_id} copy move {dx} {dy} {dz}",
            f"volume {volume_id} copy move x {dx} y {dy} z {dz}",
            f"copy volume {volume_id} move {dx} {dy} {dz}",
        ]
        for command in commands:  # try each spelling
            try:  # a failed spelling raises an error
                cubit.cmd(command)  # Cubit: copy + move
            except Exception:
                continue  # try the next spelling
            new_volumes = list(set(cubit.get_entities("volume")) - before)  # IDs created
            if len(new_volumes) == 1:  # exactly one copy
                return new_volumes[0]  # return it
            if new_volumes:  # more than one: unexpected
                # stop with a clear message
                raise RuntimeError(f'Copy of volume {volume_id} produced {len(new_volumes)} volumes')
        raise RuntimeError(f"Could not copy/move volume {volume_id}")  # no spelling worked: stop


    def assert_positive_solid(volume_id, label):  # raise if a volume is empty or a sheet body
        v = cubit.volume(volume_id)  # Cubit volume object
        if v.volume() <= 0 or any(b.is_sheet_body() for b in v.bodies()):  # zero volume or any sheet body
            raise RuntimeError(f"{label}: volume {volume_id} is not a positive solid")  # stop with a clear message


    # subtract one cutter from the RAFM, keeping the cutter; strict bookkeeping
    def subtract_and_check(rafm_id, cutter_id, stage):
        """Subtract ONE retained fluid tool from the RAFM (keep_tool)."""
        before = set(cubit.get_entities("volume"))
        if cutter_id not in before or rafm_id not in before:  # both must exist
            raise RuntimeError(f"{stage}: missing RAFM or fluid cutter")  # stop with a clear message
        # Cubit: subtract, imprint, keep the cutter
        cubit.cmd(f"subtract volume {cutter_id} from volume {rafm_id} imprint keep_tool")
        remaining = set(cubit.get_entities("volume"))  # volumes after
        if cutter_id not in remaining:  # the cutter must survive
            raise RuntimeError(f"{stage}: fluid cutter {cutter_id} was deleted")  # stop with a clear message
        if rafm_id in remaining:  # RAFM kept its ID
            answer = rafm_id  # same ID
        else:
            created = remaining - before  # RAFM got a new ID
            if len(created) != 1:  # exactly one replacement expected
                # stop with a clear message
                raise RuntimeError(f"{stage}: expected one RAFM replacement, got {sorted(created)}")
            answer = next(iter(created))  # its ID
        if remaining != (before - {rafm_id}) | {answer}:  # nothing else may change
            # stop with a clear message
            raise RuntimeError(f"{stage}: unexpected volume change {sorted(remaining ^ before)}")
        assert_positive_solid(answer, stage)  # result must be a real solid
        return answer  # RAFM ID


    def unite_retained_fluid_group(ids, label):  # unite touching fluid bodies into one solid
        """Unite fluid bodies that physically touch/overlap into ONE solid."""
        ids = list(ids)
        before = set(cubit.get_entities("volume"))  # volumes before
        if not set(ids).issubset(before):  # all inputs must exist
            raise RuntimeError(f"{label}: some retained fluid IDs are missing")  # stop with a clear message
        if len(ids) == 1:  # one body: nothing to unite
            result = ids[0]  # use it as is
        else:
            cubit.cmd("unite volume " + " ".join(map(str, ids)))  # Cubit: unite them all into one body
            after = set(cubit.get_entities("volume"))  # volumes after
            candidates = after - (before - set(ids))  # what is left of the inputs
            if len(candidates) != 1:  # must be exactly one connected solid
                # stop: some pieces do not touch
                raise RuntimeError(
                    f"{label}: unite did not form one solid, results {sorted(candidates)}. "
                    f"Some pieces do not touch."
                )
            result = next(iter(candidates))  # the united body
        assert_positive_solid(result, label)  # must be a real solid
        register_volume(result, label)  # name it
        return result  # united body ID


    # subtract the final RAFM from a fluid body (overlap check; off by default)
    def conform_fluid_to_final_rafm(fluid_id, rafm_id, label):
        """Interface check: subtract the final RAFM from a fluid body."""
        before = set(cubit.get_entities("volume"))
        original = cubit.volume(fluid_id).volume()  # fluid volume before
        # Cubit: fluid minus RAFM, keep the RAFM
        cubit.cmd(f"subtract volume {rafm_id} from volume {fluid_id} imprint keep_tool")
        after = set(cubit.get_entities("volume"))  # volumes after
        if rafm_id not in after:  # the RAFM must survive
            raise RuntimeError(f"{label}: RAFM lost during fluid trim")  # stop with a clear message
        added, removed = after - before, before - after  # what changed
        if fluid_id in after:  # fluid kept its ID
            if added or removed:  # nothing else may change
                # stop with a clear message
                raise RuntimeError(f"{label}: unexpected volumes in trim: +{added} -{removed}")
            result = fluid_id  # same ID
        else:
            if removed != {fluid_id} or len(added) != 1:  # fluid replaced by exactly one new body
                # stop with a clear message
                raise RuntimeError(f"{label}: trim split or removed fluid: +{added} -{removed}")
            result = next(iter(added))  # its ID
        assert_positive_solid(result, label)  # must be a real solid
        register_volume(result, label)  # name it
        # report how much overlap was removed
        print(f"  {label}: overlap removed {(original - cubit.volume(result).volume())*1e9:.3f} mm^3")
        return result  # fluid ID


    def level_z_bottom(level):  # Z of the bottom of the He void at a level (0 = lowest)
        """Bottom of the He void at zero-based level."""
        return IB_first_level_z + level * tube_pitch + He_bottom_wall  # first level + level pitches + bottom wall


    def level_slots(level):  # which slots a level's left / right ports feed
        """(left slot, right slot) for a zero-based level.
        Level 0 (bottom): LEFT -> middle, RIGHT -> rear; alternate upward."""
        if level % 2 == 0:  # even (0-based) levels = ODD circuit
            return MANIFOLD_SLOT_MIDDLE, MANIFOLD_SLOT_REAR  # left -> middle, right -> rear
        return MANIFOLD_SLOT_REAR, MANIFOLD_SLOT_MIDDLE  # left -> rear, right -> middle


    # ============================================================================
    # EXACT-ARC GEOMETRY (this IB-only version)
    #   Every IB outline is built from TRUE line and circular-arc curves, the
    #   same kind of exact geometry as the outboard (no polyline facets):
    #     - FW front arc, FW corner bends (R88.8 / R84.8 / R54.8 / R50.8 about one
    #       common center), side legs, U-leg end fillets (R6.35)
    #     - PbLi duct front/back arcs, straight sides and R50.8 corner fillets;
    #       SiC outer / inner surfaces with their own corner radii (reference)
    #     - He-hole R3 corners, manifold-slot corner fillets and front/back arcs,
    #       manifold rear faces, R15.24 housing blends, main back arc
    #     - port R6.35 bends
    #   Each outline is an Outline: a list of sampled points (used only by the
    #   placement / clearance checks) that also carries the exact segments
    #   (.shape) used to build the Cubit surface. The functions below REPLACE
    #   the polyline versions defined earlier (same names, same arguments).
    # ============================================================================
    class Outline(list):  # sampled points for checks + exact segments for Cubit
        def __init__(self, segs):  # segs: ("L", p0, p1) or ("A", p0, pmid, p1)
            super().__init__(sample_segments(segs))  # sampled outline points
            self.shape = segs  # exact segments

    def arc_seg(center, start, end):  # short circular arc (< 180 deg) about center, from start to end
        a0 = math.atan2(start[1] - center[1], start[0] - center[0])  # start angle
        a1 = math.atan2(end[1] - center[1], end[0] - center[0])  # end angle
        da = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi  # signed short sweep
        rad = length_xy(sub_xy(start, center))  # radius
        am = a0 + 0.5 * da  # mid angle
        mid = [center[0] + rad * math.cos(am), center[1] + rad * math.sin(am)]  # arc mid point
        return ("A", list(start), mid, list(end), list(center), rad, a0, da)  # arc record

    def line_seg(p0, p1):  # straight segment
        return ("L", list(p0), list(p1))

    def sample_segments(segs):  # points along the outline for the 2D checks (no closing duplicate)
        pts = []  # output points
        for s in segs:  # each segment
            if s[0] == "L":  # line: its start point
                pts.append(s[1])
            else:  # arc: start + interior points (<= 6 deg and <= 10 mm apart; < 0.07 mm chord error)
                c, rad, a0, da = s[4], s[5], s[6], s[7]
                n = max(2, int(math.ceil(abs(da) / math.radians(6.0))), int(math.ceil(abs(da) * rad / 0.010)))
                for k in range(n):  # start .. just before the end
                    a = a0 + da * k / n
                    pts.append([c[0] + rad * math.cos(a), c[1] + rad * math.sin(a)])
        return pts

    def check_closed(segs, label):  # consecutive segments must share end points
        for k, s in enumerate(segs):
            nxt = segs[(k + 1) % len(segs)]
            end = s[2] if s[0] == "L" else s[3]
            if length_xy(sub_xy(end, nxt[1])) > 1.0e-9:  # gap > 1 nm
                raise RuntimeError(f"{label}: outline not closed at segment {k} "
                                   f"(gap {length_xy(sub_xy(end, nxt[1]))*1e3:.6f} mm)")
        return segs

    def mirror_xy(p):  # reflect a point across the sector centerline (theta = Sector_angle / 2)
        t = math.radians(Sector_angle)  # 2 x the mirror-line angle
        return [math.cos(t) * p[0] + math.sin(t) * p[1], math.sin(t) * p[0] - math.cos(t) * p[1]]

    def mirror_segs(segs):  # mirror a segment chain AND reverse it (keeps the loop direction)
        out = []
        for s in reversed(segs):
            if s[0] == "L":
                out.append(line_seg(mirror_xy(s[2]), mirror_xy(s[1])))
            else:
                out.append(arc_seg(mirror_xy(s[4]), mirror_xy(s[3]), mirror_xy(s[1])))
        return out

    def _ARC_probe():  # make sure this Cubit can create 3-point arcs (required here)
        v = [create_vertex_xyz(*p) for p in ((0, 0, 0), (1, 1, 0), (2, 0, 0))]  # three test vertices
        c0 = cubit.get_last_id("curve")  # last curve before
        try:
            cubit.cmd(f"create curve arc three vertex {v[0]} {v[1]} {v[2]}")  # test arc
        except Exception:
            pass  # handled below
        ok = cubit.get_last_id("curve") > c0  # an arc was made
        if ok:
            cubit.cmd(f"delete curve {cubit.get_last_id('curve')}")  # removes its end vertices too
            cubit.cmd(f"delete vertex {v[1]}")  # the free middle vertex
        else:
            cubit.cmd("delete vertex " + " ".join(map(str, v)))  # clean up
            raise RuntimeError("This Cubit cannot create 3-point arcs; the exact-arc IB needs them.")
    _ARC_CHECKED = [False]  # probe once per run

    def make_surface_from_segments(segs, z_value):  # planar Cubit surface from exact segments at height z
        if not _ARC_CHECKED[0]:  # first call: probe arcs
            _ARC_probe()
            _ARC_CHECKED[0] = True
        vcache = {}  # shared end vertices
        mids = []  # arc middle vertices (deleted afterwards)

        def vid(p):  # vertex ID for an end point (created once)
            key = (round(p[0], 10), round(p[1], 10))
            if key not in vcache:
                vcache[key] = create_vertex_xyz(p[0], p[1], z_value)
            return vcache[key]
        curves = []  # boundary curve IDs
        for s in segs:  # each segment
            if s[0] == "L":
                cubit.cmd(f"create curve vertex {vid(s[1])} {vid(s[2])}")  # Cubit: straight curve
            else:
                vm = create_vertex_xyz(s[2][0], s[2][1], z_value)  # arc middle vertex
                mids.append(vm)
                cubit.cmd(f"create curve arc three vertex {vid(s[1])} {vm} {vid(s[3])}")  # Cubit: true arc
            curves.append(cubit.get_last_id("curve"))  # record it
        before = set(cubit.get_entities("surface"))  # surfaces before
        cubit.cmd("create surface curve " + " ".join(map(str, curves)))  # Cubit: planar surface
        new = list(set(cubit.get_entities("surface")) - before)  # the new surface
        if not new:
            raise RuntimeError("Exact-arc surface creation failed.")
        cubit.cmd("delete vertex " + " ".join(map(str, mids)))  # remove the free arc middle vertices
        return max(new)

    _poly_surface_vertex = make_polygon_surface_vertex  # keep the polyline builders for plain point lists
    _poly_surface_loop = make_polygon_surface_curve_loop

    def make_polygon_surface_vertex(points_xy, z_value):  # exact surface for an Outline, polyline otherwise
        if isinstance(points_xy, Outline):
            return make_surface_from_segments(points_xy.shape, z_value)
        return _poly_surface_vertex(points_xy, z_value)

    def make_polygon_surface_curve_loop(points_xy, z_value):  # exact surface for an Outline, polyline otherwise
        if isinstance(points_xy, Outline):
            return make_surface_from_segments(points_xy.shape, z_value)
        return _poly_surface_loop(points_xy, z_value)

    # --- FW / U-channel band: one common bend center per side ------------------
    _Dc = he_boundary_gap + He_outer_bend_radius  # bend center offset from the left sector wall [m]
    _Rc = IB_FW_front_radius - He_outer_bend_radius  # bend center distance from the machine axis [m]
    _bend_C = [math.sqrt(_Rc ** 2 - _Dc ** 2), _Dc]  # left bend center (common to every band offset)

    def band_left_chain(d, s_back):  # left half of the band curve at offset d: leg (from s_back) -> bend -> FW arc start
        rho = He_outer_bend_radius - (d - he_boundary_gap)  # bend radius at this offset
        R = IB_FW_front_radius - (d - he_boundary_gap)  # FW arc radius at this offset
        p_leg = [_bend_C[0], d]  # leg / bend tangent point
        p_fw = scale_xy(normalize_xy(_bend_C), R)  # bend / FW arc tangent point
        return [line_seg([s_back, d], p_leg), arc_seg(_bend_C, p_leg, p_fw)], p_fw, R

    def band_front_arc(R, p_left):  # FW arc at radius R from the left tangent point to its mirror
        return arc_seg([0.0, 0.0], p_left, mirror_xy(p_left))

    # --- U void (fluid) ---------------------------------------------------------
    def get_wrapped_he_fluid_polygon():  # EXACT U void: legs, R84.8/R54.8 bends, FW arcs, R6.35 leg-end fillets
        d1 = he_boundary_gap + He_side_wall  # void outer edge offset (4 mm)
        d2 = d1 + He_void_width  # void inner edge offset (34 mm)
        rf = He_U_fluid_end_fillet_radius  # R6.35
        R_end = He_side_leg_back_radius + He_side_wall  # leg end: arc about the axis, 4 mm in front of the rear face
        F1 = _line_circle_corner(sector_theta_min, +1, d1, R_end, -1, rf)  # outer leg line x end arc (R6.35)
        F2 = _line_circle_corner(sector_theta_min, -1, -d2, R_end, -1, rf)  # inner leg line x end arc (R6.35)
        outer, p1, R1 = band_left_chain(d1, F1[1][0])  # left outer leg + bend
        inner, p2, R2 = band_left_chain(d2, F2[1][0])  # left inner leg + bend
        left_end = [  # left leg-end cap (inner -> outer): fillet, arc about the axis, fillet
            arc_seg(F2[0], F2[1], F2[2]),
            arc_seg([0.0, 0.0], F2[2], F1[2]),
            arc_seg(F1[0], F1[2], F1[1])]
        # inner chain reversed (FW side -> leg end)
        inner_rev = [arc_seg(_bend_C, p2, [_bend_C[0], d2]), line_seg([_bend_C[0], d2], F2[1])]
        left_half = outer + [band_front_arc(R1, p1)]  # outer leg, bend, FW outer arc to the right tangent
        right_outer = mirror_segs(outer)  # right bend + right outer leg (reversed direction)
        right_end = mirror_segs(left_end)  # right leg-end cap
        right_inner = mirror_segs(inner_rev)  # right inner leg + bend (towards the FW)
        front_inner = arc_seg([0.0, 0.0], mirror_xy(p2), p2)  # FW inner arc, right -> left
        segs = left_half + right_outer + right_end + right_inner + [front_inner] + inner_rev + left_end
        return Outline(check_closed(segs, "U void"))

    # --- He port (U leg -> manifold slot) ----------------------------------------
    def port_local_flared_polygon(side_name, slot_index):  # EXACT port footprint with two true R6.35 bends
        geometry = get_manifold_side_geometry(side_name)  # side directions
        radial, inward = geometry["outer_line_direction"], geometry["outer_line_inward_normal"]
        # mouth centred on the slot's mid RADIUS (reference): distance along the wall line at the 42 mm slot face
        center_r = math.sqrt(manifold_slot_center_radius(slot_index) ** 2
                             - He_manifold_slot_outer_offset_from_sector_wall ** 2)
        hm = 0.5 * He_manifold_port_radial_length  # half mouth width
        fr = He_port_internal_fillet_radius  # R6.35
        slot_front_wall = He_manifold_slot_outer_offset_from_sector_wall  # slot face (42 mm)
        mouth_end = slot_front_wall + port_mouth_overlap(slot_index)  # mouth end inside the slot
        fillet_start = slot_front_wall - He_port_fillet_to_hole_distance  # straight throat end
        fillet_end = fillet_start - fr  # bend end
        bridge_end = He_side_fluid_inner_offset_from_sector_wall - He_port_overlap_inside_U  # tongue end in the U

        def P(s, w):  # local (s along the wall, w inward) -> XY
            return add_xy(scale_xy(radial, center_r + s), scale_xy(inward, w))
        segs = [
            line_seg(P(-hm, mouth_end), P(+hm, mouth_end)),  # mouth end (inside the slot)
            line_seg(P(+hm, mouth_end), P(+hm, fillet_start)),  # +s straight throat
            arc_seg(P(hm + fr, fillet_start), P(+hm, fillet_start), P(hm + fr, fillet_end)),  # +s R6.35 bend
            line_seg(P(hm + fr, fillet_end), P(hm + fr, bridge_end)),  # +s tongue side
            line_seg(P(hm + fr, bridge_end), P(-hm - fr, bridge_end)),  # tongue end (inside the U void)
            line_seg(P(-hm - fr, bridge_end), P(-hm - fr, fillet_end)),  # -s tongue side
            arc_seg(P(-hm - fr, fillet_start), P(-hm - fr, fillet_end), P(-hm, fillet_start)),  # -s R6.35 bend
            line_seg(P(-hm, fillet_start), P(-hm, mouth_end))]  # -s straight throat
        return Outline(check_closed(segs, f"port {side_name} {slot_index}"))

    # --- manifold slot -------------------------------------------------------------
    def make_side_manifold_opening_polygon(side_name, slot_index):  # EXACT slot: 2 lines, 2 origin arcs, 4 fillets
        data = get_slot_line_data(side_name)  # bounding lines
        fr_, br_ = manifold_slot_front_radius(slot_index), manifold_slot_back_radius(slot_index)
        r_port = manifold_slot_corner_radius(slot_index)  # port-side corners (the slot test may reduce them)
        r_wall = Manifold_slot_fillet_radius  # housing-wall-side corners: always the design radius
        F = lambda dirk, nk, ok, R, front, r: line_circle_fillet_data(data[dirk], data[nk], data[ok], R, r,
                                                                     circle_is_front=front)
        of = F("outer_direction", "outer_interior_normal", "outer_offset", fr_, True, r_port)  # port side, front
        ob = F("outer_direction", "outer_interior_normal", "outer_offset", br_, False, r_port)  # port side, back
        ib = F("inner_direction", "inner_interior_normal", "inner_offset", br_, False, r_wall)  # wall side, back
        if_ = F("inner_direction", "inner_interior_normal", "inner_offset", fr_, True, r_wall)  # wall side, front
        O = [0.0, 0.0]
        segs = [line_seg(of["line_tangent"], ob["line_tangent"]),  # outer straight edge
                arc_seg(ob["center"], ob["line_tangent"], ob["circle_tangent"]),  # outer/back fillet
                arc_seg(O, ob["circle_tangent"], ib["circle_tangent"]),  # back arc
                arc_seg(ib["center"], ib["circle_tangent"], ib["line_tangent"]),  # inner/back fillet
                line_seg(ib["line_tangent"], if_["line_tangent"]),  # inner straight edge
                arc_seg(if_["center"], if_["line_tangent"], if_["circle_tangent"]),  # inner/front fillet
                arc_seg(O, if_["circle_tangent"], of["circle_tangent"]),  # front arc
                arc_seg(of["center"], of["circle_tangent"], of["line_tangent"])]  # outer/front fillet
        return Outline(check_closed(segs, f"slot {side_name} {slot_index}"))

    # --- IB RAFM body -----------------------------------------------------------------
    def housing_blend(side_name):  # rear-inner corner, wall tangent, fillet center, back-circle tangent (one side)
        g = get_manifold_side_geometry(side_name)
        back = He_manifold_back_axis_coordinate  # manifold rear face radius
        r = He_manifold_to_RAFM_inner_fillet_radius  # R15.24 blend
        Rb = He_main_RAFM_back_edge_radius  # main back arc radius
        u, nrm, off = g["inner_line_direction"], g["inner_line_outward_normal"], g["inner_wall_offset"]  # wall line
        depth = math.sqrt((Rb - r) ** 2 - (off - r) ** 2)  # fillet center along the wall direction
        C = add_xy(scale_xy(u, depth), scale_xy(nrm, off - r))  # fillet center (cavity side of the wall)
        wall_t = add_xy(scale_xy(u, depth), scale_xy(nrm, off))  # tangent on the housing wall
        circ_t = scale_xy(C, Rb / (Rb - r))  # tangent on the main back circle
        rear_inner = add_xy(scale_xy(u, math.sqrt(back ** 2 - off ** 2)), scale_xy(nrm, off))  # wall at the rear face
        return rear_inner, wall_t, C, circ_t

    def make_curved_blanket_body_polygon():  # EXACT IB slab outline
        back = He_manifold_back_axis_coordinate  # legs run back to the manifold rear face
        outer, p0, R0_ = band_left_chain(he_boundary_gap, back)  # left wall leg + R88.8 bend
        L_in, L_wt, L_C, L_ct = housing_blend("left")
        R_in, R_wt, R_C, R_ct = housing_blend("right")
        segs = (outer + [band_front_arc(R0_, p0)] + mirror_segs(outer)  # wall legs, bends, FW arc
                + [arc_seg([0.0, 0.0], mirror_xy([back, he_boundary_gap]), R_in),  # right manifold rear face
                   line_seg(R_in, R_wt),  # right manifold inner wall
                   arc_seg(R_C, R_wt, R_ct),  # right R15.24 blend
                   arc_seg([0.0, 0.0], R_ct, L_ct),  # main back arc
                   arc_seg(L_C, L_ct, L_wt),  # left R15.24 blend
                   line_seg(L_wt, L_in),  # left manifold inner wall
                   arc_seg([0.0, 0.0], L_in, [back, he_boundary_gap])])  # left manifold rear face
        return Outline(check_closed(segs, "IB body"))

    # --- PbLi ducts and SiC layers ----------------------------------------------------------
    #   Duct sides are straight lines PARALLEL to a radial line: half a column
    #   gap from a dividing radial line (so every web between ducts has a
    #   constant thickness), or He band thickness from the sector wall (outer
    #   side of the edge ducts). Front and back faces are arcs about the axis.
    #   A side line is stored as (theta, c): the radial line at angle theta,
    #   moved sideways by c toward the duct interior.
    def _line_circle_corner(theta, n_sign, c, R, sigma, rho):  # fillet between a side line and an origin circle
        u = radial_unit(theta)  # line direction (outward)
        n = scale_xy(tangent_unit(theta), n_sign)  # interior normal of the side line
        t = math.sqrt((R - sigma * rho) ** 2 - (c + rho) ** 2)  # fillet center position along the line
        C = add_xy(scale_xy(n, c + rho), scale_xy(u, t))  # fillet center
        line_t = add_xy(scale_xy(n, c), scale_xy(u, t))  # tangent point on the side line
        circ_t = scale_xy(C, R / length_xy(C))  # tangent point on the circle
        return C, line_t, circ_t  # center, line tangent, circle tangent

    def duct_from_lines(r_front, r_back, left, right, rho, label):  # EXACT rounded outline: 2 side lines + 2 arcs
        (tl, cl), (tr, cr) = left, right  # side lines (theta, offset)
        LO = _line_circle_corner(tl, +1, cl, r_front, +1, rho)  # left side x front arc
        LI = _line_circle_corner(tl, +1, cl, r_back, -1, rho)  # left side x back arc
        RO = _line_circle_corner(tr, -1, cr, r_front, +1, rho)  # right side x front arc
        RI = _line_circle_corner(tr, -1, cr, r_back, -1, rho)  # right side x back arc
        if length_xy(LI[1]) >= length_xy(LO[1]) or length_xy(RI[1]) >= length_xy(RO[1]):  # sides must keep a length
            raise ValueError(f"{label}: corner radius {rho*1000:.2f} mm does not fit its length.")
        if math.atan2(RO[2][1], RO[2][0]) <= math.atan2(LO[2][1], LO[2][0]):  # front arc must keep a length
            raise ValueError(f"{label}: corner radius {rho*1000:.2f} mm does not fit its width.")
        if math.atan2(RI[2][1], RI[2][0]) <= math.atan2(LI[2][1], LI[2][0]):  # back arc must keep a length
            raise ValueError(f"{label}: corner radius {rho*1000:.2f} mm does not fit its width.")
        O = [0.0, 0.0]  # machine axis
        segs = [line_seg(LI[1], LO[1]),  # left side
                arc_seg(LO[0], LO[1], LO[2]),  # left-front fillet
                arc_seg(O, LO[2], RO[2]),  # front arc
                arc_seg(RO[0], RO[2], RO[1]),  # right-front fillet
                line_seg(RO[1], RI[1]),  # right side
                arc_seg(RI[0], RI[1], RI[2]),  # right-back fillet
                arc_seg(O, RI[2], LI[2]),  # back arc
                arc_seg(LI[0], LI[2], LI[1])]  # left-back fillet
        return Outline(check_closed(segs, label))  # exact outline

    def side_midpoint(line, side, r_front, r_back, rho):  # midpoint of the STRAIGHT part of a duct side
        theta, c = line  # side line
        t_f = math.sqrt((r_front - rho) ** 2 - (c + rho) ** 2)  # front fillet tangent position along the line
        t_b = math.sqrt((r_back + rho) ** 2 - (c + rho) ** 2)  # back fillet tangent position along the line
        n = scale_xy(tangent_unit(theta), +1.0 if side == "left" else -1.0)  # interior normal
        return add_xy(scale_xy(n, c), scale_xy(radial_unit(theta), 0.5 * (t_f + t_b)))  # midpoint

    def duct_width(left, right, r_front, r_back, rho):  # width = distance between the two side midpoints
        return length_xy(sub_xy(side_midpoint(left, "left", r_front, r_back, rho),
                                side_midpoint(right, "right", r_front, r_back, rho)))

    def get_PbLi_row_radial_bounds():  # (front, back) radius of every PbLi row
        return list(PbLi_row_bounds)

    _ROW_LAYOUT = {}  # cache: row -> column layout

    def row_column_layout(row_index):  # columns of one row: Num_PbLi_cols_IB equal wedges
        if row_index in _ROW_LAYOUT:
            return _ROW_LAYOUT[row_index]
        r_front, r_back = PbLi_row_bounds[row_index]  # row radii
        rho = PbLi_corner_radius  # duct corner radius
        hg = 0.5 * PbLi_intercolumn_web_width  # half a web
        n_cols = int(Num_PbLi_cols_IB)  # columns (input)
        if n_cols < 3:
            raise ValueError("Num_PbLi_cols_IB must be at least 3 (two edge ducts + one inner duct).")
        alpha = Sector_angle / n_cols  # equal wedge angle [deg]
        n_inner = n_cols - 2  # inner ducts
        dividers = [sector_theta_min + k * alpha for k in range(1, n_cols)]  # wedge lines [deg]
        wall = PbLi_side_edge_offset_from_sector_boundary  # edge duct outer side offset from the wall [m] (38 mm)

        def inner_width(a_):  # width of an inner duct whose wedge lines are a_ deg apart
            mid = 0.5 * (sector_theta_min + sector_theta_max)
            return duct_width((mid - 0.5 * a_, hg), (mid + 0.5 * a_, hg), r_front, r_back, rho)
        try:
            inner_w = inner_width(alpha)  # inner duct width [m]
            edge_w = duct_width((sector_theta_min, wall), (dividers[0], hg), r_front, r_back, rho)  # edge width [m]
        except ValueError:
            raise ValueError(f"PbLi row {row_index + 1}: {n_cols} columns are too many for the sector "
                             f"(a duct and its {rho*1000:.1f} mm corners do not fit a {alpha:.4f} deg wedge).")
        if min(inner_w, edge_w) <= 2.0 * rho:
            raise ValueError(f"PbLi row {row_index + 1}: with {n_cols} columns a duct is only "
                             f"{min(inner_w, edge_w)*1000:.2f} mm wide (less than two {rho*1000:.1f} mm corner fillets).")
        lines = [((sector_theta_min, wall), (dividers[0], hg))]  # column 1 (edge)
        lines += [((dividers[k], hg), (dividers[k + 1], hg)) for k in range(n_inner)]  # inner columns
        lines += [((dividers[-1], hg), (sector_theta_max, wall))]  # last column (edge)
        lay = {"alpha": alpha, "dividers": dividers, "lines": lines, "n_cols": n_cols,
               "inner_width": inner_w, "edge_width": edge_w}
        _ROW_LAYOUT[row_index] = lay
        return lay

    def get_PbLi_layout():  # all IB PbLi ducts: radii, side lines, exact outlines
        layout = []  # one dict per duct
        for row_index, (r_outer, r_inner) in enumerate(get_PbLi_row_radial_bounds()):  # each row
            lay = row_column_layout(row_index)  # its columns
            r_mid = 0.5 * (r_outer + r_inner)  # row mid radius
            for col_index, (left, right) in enumerate(lay["lines"]):  # each column
                th0 = side_line_theta_at(left, r_mid, "left")  # left side angle at mid radius
                th1 = side_line_theta_at(right, r_mid, "right")  # right side angle at mid radius
                label = f"PbLi duct R{row_index + 1} C{col_index + 1}"  # label for messages
                polygon = duct_from_lines(r_outer, r_inner, left, right, PbLi_corner_radius, label)  # exact outline
                layout.append({"row_index": row_index, "col_index": col_index, "n_cols": lay["n_cols"],
                               "r_outer": r_outer, "r_inner": r_inner, "r_mid": r_mid,
                               "theta_min": th0, "theta_max": th1, "theta_mid": 0.5 * (th0 + th1),
                               "left": left, "right": right, "polygon": polygon})
        return layout  # list of ducts

    def side_line_theta_at(line, radius, side):  # polar angle [deg] where a side line crosses a radius
        theta, c = line  # line angle and offset
        d = math.degrees(math.asin(c / radius))  # angular shift of the offset line at this radius
        return theta + d if side == "left" else theta - d  # left lines move to +theta, right lines to -theta

    def make_inset_PbLi_polygon(duct, inset):  # SiC outer (inset = gap) or SiC inner (inset = gap + wall) surface
        row, col = duct["row_index"], duct["col_index"]  # duct position
        front_gap = SiC_front_gap_by_row[row]  # PbLi gap at the front face of this row
        edge = col in (0, duct["n_cols"] - 1)  # edge duct?
        if abs(inset - SiC_offset_from_PbLi_cavity) < 1.0e-12:  # SiC OUTER surface
            front, side, rho, tag = front_gap, SiC_offset_from_PbLi_cavity, SiC_outer_corner_radius, "SiC outer"
        elif abs(inset - SiC_total_inset_to_inner_fluid) < 1.0e-12:  # SiC INNER surface (PbLi core)
            front, side, tag = front_gap + SiC_wall_thickness, SiC_total_inset_to_inner_fluid, "SiC inner"
            rho = SiC_inner_corner_radius_edge if edge else SiC_inner_corner_radius_center  # its corner radius
        else:
            raise ValueError(f"Unexpected SiC inset {inset * 1000.0:.3f} mm.")  # only the two SiC surfaces exist
        (tl, cl), (tr, cr) = duct["left"], duct["right"]  # cavity side lines
        return duct_from_lines(duct["r_outer"] - front, duct["r_inner"] + side,  # front / back arcs moved in
                               (tl, cl + side), (tr, cr + side), rho,  # sides moved in
                               f"{tag} R{row + 1} C{col + 1}")

    # --- manifold housing inner wall: radial line, same angle from both sector walls ---------------------------
    if Manifold_housing_wall_angle is None:  # derived: radial line through duct 2's extended outer side at the rear face
        _last = row_column_layout(num_rows - 1)  # last-row columns
        _first_divider = 0.5 * ((_last["dividers"][0] - sector_theta_min)
                                + (sector_theta_max - _last["dividers"][-1]))  # first dividing line (symmetric mean) [deg]
        Housing_wall_angle_used = _first_divider + math.degrees(
            math.asin(0.5 * PbLi_intercolumn_web_width / He_manifold_back_axis_coordinate))  # [deg]
    else:  # user override
        Housing_wall_angle_used = float(Manifold_housing_wall_angle)  # [deg]
    He_manifold_left_inner_wall_theta = sector_theta_min + Housing_wall_angle_used  # left wall angle [deg]
    He_manifold_right_inner_wall_theta = sector_theta_max - Housing_wall_angle_used  # right wall angle [deg] (mirror)
    He_manifold_inner_wall_line_offset = 0.0  # the wall is a radial line (no sideways offset)

    # --- internal He holes ---------------------------------------------------------------
    def make_oriented_rounded_rect_polygon(center_radius, center_theta, tangential_width, radial_height, corner_radius):
        rho = min(corner_radius, 0.49 * tangential_width, 0.49 * radial_height)  # corner radius that fits
        t, rv = tangent_unit(center_theta), radial_unit(center_theta)  # local axes
        c0 = point_xy(center_radius, center_theta)  # hole center
        hw, hh = 0.5 * tangential_width, 0.5 * radial_height  # half sizes
        P = lambda a_, b_: add_xy(c0, add_xy(scale_xy(t, a_), scale_xy(rv, b_)))  # local -> XY
        segs = [line_seg(P(-hw + rho, -hh), P(hw - rho, -hh)),  # back edge
                arc_seg(P(hw - rho, -hh + rho), P(hw - rho, -hh), P(hw, -hh + rho)),  # corner
                line_seg(P(hw, -hh + rho), P(hw, hh - rho)),  # +t side
                arc_seg(P(hw - rho, hh - rho), P(hw, hh - rho), P(hw - rho, hh)),  # corner
                line_seg(P(hw - rho, hh), P(-hw + rho, hh)),  # front edge
                arc_seg(P(-hw + rho, hh - rho), P(-hw + rho, hh), P(-hw, hh - rho)),  # corner
                line_seg(P(-hw, hh - rho), P(-hw, -hh + rho)),  # -t side
                arc_seg(P(-hw + rho, -hh + rho), P(-hw, -hh + rho), P(-hw + rho, -hh))]  # corner
        return Outline(check_closed(segs, "He hole"))  # exact outline

    def sector_hole_polygon(theta_c, r_in, r_out, half_angle, rho):  # arc-row hole: radial sides, arc faces
        return duct_from_lines(r_out, r_in, (theta_c - half_angle, 0.0), (theta_c + half_angle, 0.0), rho, "He hole")

    def check_ligament(value, label):  # stop if a RAFM ligament is thinner than the design minimum
        if value < Min_RAFM_ligament - 1.0e-12:
            raise ValueError(f"{label}: RAFM ligament {value*1000:.3f} mm < Min_RAFM_ligament "
                             f"{Min_RAFM_ligament*1000:.3f} mm.")

    def arc_row_plan(r_in, r_out, width, rib, label):  # holes along an arc: fixed rib, as many as fit, centred
        r_mid = 0.5 * (r_in + r_out)  # hole mid radius (width and rib are measured here)
        lim = math.degrees(math.asin(PbLi_side_edge_offset_from_sector_boundary / r_mid))  # He band inner faces
        avail = math.radians(Sector_angle - 2.0 * lim) * r_mid  # usable arc between the bands [m]
        n = int(math.floor((avail + rib) / (width + rib) + 1.0e-9))  # holes that fit with the fixed rib
        while n > 0 and 0.5 * (avail - n * width - (n - 1) * rib) < Min_RAFM_ligament - 1.0e-12:
            n -= 1  # keep at least Min_RAFM_ligament at both ends
        if n < 1:
            raise ValueError(f"{label}: no hole fits.")
        check_ligament(rib * r_in / r_mid, f"{label} rib at the inner radius")  # narrowest rib
        end = 0.5 * (avail - n * width - (n - 1) * rib)  # leftover at each end [m]
        hole_ang, rib_ang = math.degrees(width / r_mid), math.degrees(rib / r_mid)  # as angles
        start = 0.5 * (sector_theta_min + sector_theta_max) - 0.5 * (n * hole_ang + (n - 1) * rib_ang)  # centred
        angles = [start + 0.5 * hole_ang + k * (hole_ang + rib_ang) for k in range(n)]  # hole centre angles
        return {"label": label, "r_in": r_in, "r_out": r_out, "angles": angles, "half": 0.5 * hole_ang,
                "width": width, "rib": rib, "end": end}

    def large_He_rows():  # every row-divider hole row + the rear row
        rows = []
        for i in range(num_rows - 1):  # one row in each gap between duct rows, centred in the gap
            gap_front, gap_back = PbLi_row_bounds[i][1], PbLi_row_bounds[i + 1][0]  # gap radii
            lig = 0.5 * ((gap_front - gap_back) - Divider_He_hole_length)  # RAFM each side of the holes
            check_ligament(lig, f"DIVIDER He row {i + 1}")
            r = arc_row_plan(gap_back + lig, gap_front - lig, Divider_He_hole_width, Divider_He_hole_rib,
                             f"DIVIDER He row {i + 1}")
            r["corner"] = Divider_He_hole_fillet_radius
            rows.append(r)
        check_ligament(Rear_He_gap_to_ducts, "REAR He row -> ducts")
        check_ligament(Rear_He_gap_to_back_wall, "REAR He row -> back wall")
        r = arc_row_plan(rear_He_row_back_radius, rear_He_row_front_radius, Rear_He_hole_width, Rear_He_hole_rib,
                         "REAR He row")
        r["corner"] = Divider_He_hole_fillet_radius
        rows.append(r)
        return rows

    def column_hole_plan(r_back, r_front, label):  # centre radii of the holes along one duct row: fixed rib, centred
        L, h, g = r_front - r_back, Column_He_hole_length, Column_He_hole_rib  # row length, hole, rib
        n = int(math.floor((L + g) / (h + g) + 1.0e-9))  # holes that fit
        while n > 0 and 0.5 * (L - n * h - (n - 1) * g) < Min_RAFM_ligament - 1.0e-12:
            n -= 1  # keep at least Min_RAFM_ligament at both ends
        if n < 1:
            return [], 0.0
        end = 0.5 * (L - n * h - (n - 1) * g)  # leftover at each end [m]
        return [r_back + end + 0.5 * h + k * (h + g) for k in range(n)], end

    def small_web_holes():  # every column hole: (row, web, centre radius, web angle)
        holes = []
        for row_index, (r_outer, r_inner) in enumerate(PbLi_row_bounds):  # each row
            radii, _end = column_hole_plan(r_inner, r_outer, f"COLUMN He row {row_index + 1}")
            for w, theta in enumerate(row_column_layout(row_index)["dividers"]):  # each web centre line
                for rc in radii:
                    holes.append((row_index, w, rc, theta))
        return holes

    def cut_hole_group(blanket_body_id, body_polygon, pbli_layout, polys, copy_fn, label, progress_state):
        # check every hole, build ONE prototype prism, make the others with copy_fn, cut them all (kept as He)
        ok = [k for k, p in enumerate(polys) if candidate_hole_is_safe(p, body_polygon, pbli_layout)]  # safe holes
        if len(ok) < len(polys):  # some failed the clearance checks
            print(f"  {label}: {len(polys) - len(ok)} hole(s) skipped by the safety checks")
            progress_state["total"] -= len(polys) - len(ok)  # remove them from the progress total
        if not ok:  # nothing to cut
            return blanket_body_id, [], []
        proto = make_prism_from_polygon_curve_loop(polys[ok[0]], cut_z_min, cut_z_max)  # prototype prism
        ids = [proto] + [copy_fn(proto, ok[0], k) for k in ok[1:]]  # copies
        blanket_body_id = subtract_cutter_batch_from_body(blanket_body_id, ids, label,  # batched subtract
                                                          He_hole_boolean_batch_size, progress_state=progress_state)
        return blanket_body_id, [polys[k] for k in ok], ids  # body, cut outlines, He bodies

    def cut_patterned_he_holes(blanket_body_id, body_polygon, pbli_layout):  # all IB internal He holes
        rows = large_He_rows()  # divider rows + rear row
        small = small_web_holes()  # column holes
        total = sum(len(r["angles"]) for r in rows) + len(small)  # holes planned
        progress_state = {"completed": 0, "total": total}  # running progress counter
        print("")
        print(f"IB internal He-hole cutters planned = {total}")
        all_polys, all_ids = [], []  # cut outlines and He bodies
        for r in rows:  # arc rows: one prototype + rotate copies
            polys = [sector_hole_polygon(a_, r["r_in"], r["r_out"], r["half"], r["corner"]) for a_ in r["angles"]]
            rot = lambda pid, k0, k, A=r["angles"]: copy_volume_rotate_z(pid, A[k] - A[k0])  # rotate copy
            blanket_body_id, p, i = cut_hole_group(blanket_body_id, body_polygon, pbli_layout, polys, rot,
                                                   r["label"], progress_state)
            all_polys += p; all_ids += i  # collect
        groups = {}  # column holes grouped per (row, web): one prototype + translate copies
        for (row_index, w, rc, theta) in small:
            groups.setdefault((row_index, w), []).append((rc, theta))
        for (row_index, w), members in sorted(groups.items()):
            theta = members[0][1]  # web angle
            polys = [make_oriented_rounded_rect_polygon(rc, theta, Column_He_hole_width, Column_He_hole_length,
                                                        Column_He_hole_fillet_radius) for rc, _t in members]
            u = radial_unit(theta)  # web direction
            mv = lambda pid, k0, k, M=members, u=u: copy_volume_move_xy(pid, (M[k][0] - M[k0][0]) * u[0],
                                                                          (M[k][0] - M[k0][0]) * u[1])  # translate
            blanket_body_id, p, i = cut_hole_group(blanket_body_id, body_polygon, pbli_layout, polys, mv,
                                                   f"COLUMN He row {row_index + 1} web {w + 1}", progress_state)
            all_polys += p; all_ids += i  # collect
        print("")
        print(f"TOTAL SUCCESSFUL He HOLES = {len(all_polys)}")
        return (blanket_body_id, len(all_polys), all_polys, all_ids)  # same return as before

    def validate_primary_geometry():  # stop early on impossible inputs; print the calculated layout
        if Num_sectors < 3:
            raise ValueError("Num_sectors must be at least 3.")
        if num_rows < 1:
            raise ValueError("Num_PbLi_rows_IB must be at least 1.")
        if IB_body_back_radius <= 0.0 or He_side_leg_back_radius <= 0.0:
            raise ValueError("The inboard radial build reaches the machine axis.")
        if Number_of_He_tubes < 1:
            raise ValueError("The inboard is too short for one wrapped-He level.")
        if He_manifold_port_radial_length >= He_manifold_slot_radial_length:
            raise ValueError("Manifold_port_width must be smaller than Manifold_slot_length.")
        if He_port_internal_fillet_radius >= 0.5 * min(He_manifold_port_radial_length, He_manifold_port_height):
            raise ValueError("He_channel_end_fillet_radius must be < half of both port opening sizes.")
        if not (0.0 < He_U_fluid_end_fillet_radius < 0.5 * He_void_width):
            raise ValueError("He_channel_end_fillet_radius must fit the He channel width.")
        if straight_side_leg_length <= 0.0:
            raise ValueError("Wrapped-He side leg is too short for the FW corner radius.")
        for v, name in ((He_channel_wall, "He_channel_wall"), (0.5 * He_channel_stack_wall, "He_channel_stack_wall / 2"),
                        (He_manifold_gap_from_wrapped_channel, "Manifold_gap_to_wrapped_channel"),
                        (He_manifold_inter_slot_rib, "Manifold_slot_gap"),
                        (Manifold_back_wall_to_first_slot, "Manifold_back_wall_to_first_slot"),
                        (0.5 * (PbLi_intercolumn_web_width - Column_He_hole_width), "COLUMN He holes side ligament"),
                        (Column_He_hole_rib, "Column_He_hole_rib"), (Divider_He_hole_rib, "Divider_He_hole_rib"),
                        (Rear_He_hole_rib, "Rear_He_hole_rib")):
            check_ligament(v, name)
        for duct in get_PbLi_layout():  # every duct and SiC outline (raises if a corner does not fit)
            make_inset_PbLi_polygon(duct, SiC_offset_from_PbLi_cavity)
            make_inset_PbLi_polygon(duct, SiC_total_inset_to_inner_fluid)
        rows = large_He_rows()  # raises if a hole row does not fit
        print("")
        print("INBOARD LAYOUT (calculated from the user inputs)")
        print(f"  sector angle {Sector_angle:.4f} deg, FW radius {IB_FW_front_radius*1000:.3f} mm, "
              f"RAFM back wall {IB_body_back_radius*1000:.3f} mm, manifold rear face {He_manifold_back_axis_coordinate*1000:.3f} mm")
        for i, (rf, rb) in enumerate(PbLi_row_bounds):
            lay = row_column_layout(i)
            radii, end = column_hole_plan(rb, rf, "")
            print(f"  PbLi row {i+1}: {rf*1000:.3f} -> {rb*1000:.3f} mm, {lay['n_cols']} columns "
                  f"(inner {lay['inner_width']*1000:.2f} mm, edge {lay['edge_width']*1000:.2f} mm = "
                  f"{lay['edge_width']/lay['inner_width']*100:.1f} %, pitch {lay['alpha']:.5f} deg); "
                  f"{len(radii)} column holes per web, {end*1000:.3f} mm at each end")
        for r in rows:
            print(f"  {r['label']}: {len(r['angles'])} holes, {(r['r_out']-r['r_in'])*1000:.2f} x {r['width']*1000:.2f} mm, "
                  f"rib {r['rib']*1000:.2f} mm, {r['end']*1000:.3f} mm left at each end")
        print(f"  manifold housing wall: radial line {Housing_wall_angle_used:.4f} deg from each sector wall"
              + (" (derived)" if Manifold_housing_wall_angle is None else " (input)"))
        data = get_slot_line_data("left")  # slot side lines
        for k in range(3):  # slot width along the arc at its mid radius
            Rm = manifold_slot_center_radius(k)
            po = point_on_offset_line_at_radius(data["outer_direction"], data["outer_interior_normal"], data["outer_offset"], Rm)
            pi_ = point_on_offset_line_at_radius(data["inner_direction"], data["inner_interior_normal"], data["inner_offset"], Rm)
            w = Rm * abs(math.atan2(po[1], po[0]) - math.atan2(pi_[1], pi_[0]))
            print(f"  manifold slot {k+1}: width {w*1000:.2f} mm at r = {Rm*1000:.3f} mm "
                  f"(corner R{manifold_slot_corner_radius(k)*1000:.3f})")
        for side_name in ("left", "right"):  # every slot outline (raises if a fillet does not fit)
            for slot_index in range(3):
                make_side_manifold_opening_polygon(side_name, slot_index)


    def stage(msg):  # log an inboard stage and (optionally) redraw
        log("IB: " + msg)  # time-stamped message
        show()  # redraw if graphics are on


    # ============================================================================
    # MAIN BUILD
    # ============================================================================

    N = Number_of_He_tubes  # number of wrapped-He levels

    if He_network_boolean_mode not in ("sequential", "united"):  # only two network modes exist
        raise ValueError("He_network_boolean_mode must be 'sequential' or 'united'")  # stop with a clear message

    print("\n==============================================================")  # banner
    # IB summary line
    print(f" FNSF DCLL INBOARD: {N} wrapped-He levels in {IB_height*1000:.2f} mm "
          f"(extra RAFM {0.5*IB_extra_height*1000:.2f} mm top and bottom), "
          f"Z {IB_z_bottom:.4f}..{IB_z_top:.4f} m, mode = {He_network_boolean_mode}")
    print("==============================================================")  # banner

    validate_primary_geometry()  # stop early on impossible inputs

    stage("RAFM body")  # stage message
    body_polygon = make_curved_blanket_body_polygon()  # plan outline of the IB RAFM slab
    rafm = make_blanket_body_volume(body_polygon)  # extrude it over the full height

    # ---------------------------------------------------------------------------
    # SLOT CORNER TEST (before anything is cut): for the middle and rear slots,
    # build a short piece of the real network around one port on each side
    # (U void + port + slot, united; then cut from a short RAFM slice) and
    # ACIS-validate it. Keep the first corner radius that passes.
    # ---------------------------------------------------------------------------
    _err_count = getattr(cubit, "get_error_count", None)  # Cubit error counter (may not exist)

    def ib_volume_is_valid(vid):  # True/False from 'validate volume', None if it cannot be told
        if not callable(_err_count):  # counter not available in this Cubit
            cubit.cmd(f"validate volume {vid}")  # still run it (output in the console)
            return None  # unknown
        n0 = _err_count()  # errors before
        cubit.cmd(f"validate volume {vid}")  # Cubit: ACIS check
        return _err_count() == n0  # valid if no new errors

    def mouth_crossing_angle(slot_index, radius):  # angle [deg] where a port side meets the slot outline
        straight_half = 0.5 * He_manifold_slot_radial_length - radius  # half length of the straight edge
        u = 0.5 * He_manifold_port_radial_length - straight_half  # how far the mouth side reaches into the arc
        return 90.0 if u <= 0.0 else math.degrees(math.asin(min(1.0, u / radius)))  # 90 = on the straight edge

    def slot_test_piece_ok(slot_index):  # build + validate test pieces for this slot (both sides)
        ok_all = True  # stays True unless a check fails
        for side in ("left", "right"):  # both manifolds
            parity = 0 if level_slots(0)[0 if side == "left" else 1] == slot_index else 1  # level that feeds it
            if parity >= N:  # only one level exists
                continue  # this side never feeds this slot
            zb = level_z_bottom(parity)  # void bottom at that level
            z0, z1 = zb - tube_pitch, zb + He_void_height + tube_pitch  # short Z window around the port
            u_t = make_prism_from_polygon_vertex(get_wrapped_he_fluid_polygon(), zb, zb + He_void_height)  # U void
            p_t = make_prism_from_polygon_vertex(port_local_flared_polygon(side, slot_index), zb,
                                                 zb + He_manifold_port_height)  # port
            s_t = make_prism_from_polygon_curve_loop(make_side_manifold_opening_polygon(side, slot_index),
                                                     z0, z1)  # short slot piece
            fluid = unite_retained_fluid_group([u_t, p_t, s_t], f"slot test {side} {slot_index+1}")  # united network
            body = make_prism_from_polygon_vertex(body_polygon, z0, z1)  # short RAFM slice
            body = subtract_and_check(body, fluid, f"slot test cut {side} {slot_index+1}")  # cut it (keep fluid)
            for vid in (fluid, body):  # check both results
                if ib_volume_is_valid(vid) is False:  # ACIS found a problem
                    ok_all = False  # this radius fails
            cubit.cmd(f"delete volume {fluid} {body}")  # remove the test pieces
        return ok_all  # True if every piece passed

    stage("Manifold slot corner test (middle + rear slots)")  # stage message
    R_fit = (0.5 * He_manifold_slot_radial_length - 0.5 * He_manifold_port_radial_length
             - Slot_mouth_clearance_to_arc)  # largest radius with a clean mouth crossing [m]
    if R_fit <= 0.0:  # the mouth does not fit on the slot at all
        raise ValueError("Port mouth + clearance is longer than the manifold slot.")  # stop
    for slot in (MANIFOLD_SLOT_MIDDLE, MANIFOLD_SLOT_REAR):  # the slots that receive ports
        name = "middle" if slot == MANIFOLD_SLOT_MIDDLE else "rear"  # label
        design = Slot_corner_radius_used[slot]  # design radius from the inputs
        start = design if design <= R_fit else R_fit  # design if it fits the rule, else the derived fit
        if start < design:  # design radius is near-tangent to the port mouth
            log(f"  {name} slot: design R{design*1000:.2f} puts the port mouth "
                f"{mouth_crossing_angle(slot, design):.1f} deg onto the corner arc (near-tangent); "
                f"largest clean radius = R{R_fit*1000:.3f}")  # explain
        tries = [start - k * Slot_radius_step for k in range(Slot_radius_max_steps + 1)]  # largest first
        tries = [r for r in tries if r > 0.0]  # positive radii only
        for k, radius in enumerate(tries):  # each candidate radius
            Slot_corner_radius_used[slot] = radius  # use it for the outlines
            ok = slot_test_piece_ok(slot)  # build + validate the test pieces
            if ok:  # passed
                log(f"  {name} slot: using R{radius*1000:.3f} (mouth crossing "
                    f"{mouth_crossing_angle(slot, radius):.0f} deg, test pieces valid)")  # report
                break  # keep this radius
            if k == len(tries) - 1:  # nothing left to try
                log(f"  WARNING {name} slot: R{radius*1000:.3f} still fails the test (kept; check it)")  # warn
            else:  # try a smaller radius
                log(f"  {name} slot R{radius*1000:.3f}: test pieces failed ACIS validate -> trying "
                    f"R{tries[k+1]*1000:.3f}")  # report

    # ---------------------------------------------------------------------------
    # PHASE 1 (A). Wrapped-He network: slot prisms, U + port prototypes, 2-level
    #             unit cell, Z copies, ODD / EVEN circuits, cut into the plain body
    # ---------------------------------------------------------------------------
    stage("Manifold slot tools (6 full-height prisms)")
    manifold_tools = {}  # (side, slot) -> slot prism ID
    for side in ("left", "right"):  # both sides
        for slot in (MANIFOLD_SLOT_FRONT, MANIFOLD_SLOT_MIDDLE, MANIFOLD_SLOT_REAR):  # all three slots
            outline = make_side_manifold_opening_polygon(side, slot)  # slot outline
            # full-height slot prism (open top and bottom)
            manifold_tools[side, slot] = make_prism_from_polygon_curve_loop(outline, IB_z_bottom, IB_z_top)

    stage("Wrapped-He U prototype + port prototypes")  # stage message
    u_shape = get_wrapped_he_fluid_polygon()  # U void outline
    # U prism at level 0
    u_proto = make_prism_from_polygon_vertex(u_shape, level_z_bottom(0), level_z_bottom(0) + He_void_height)

    # One port prototype per (level parity, side), built at level 0 or 1.
    port_proto = {}
    for parity in (0, 1):  # level 0 and level 1 prototypes
        if parity >= N:  # only one level: no level-1 ports
            continue  # skip
        left_slot, right_slot = level_slots(parity)  # slots fed at this parity
        zb = level_z_bottom(parity)  # void bottom Z at this level
        for side, slot in (("left", left_slot), ("right", right_slot)):  # both ports of this level
            footprint = port_local_flared_polygon(side, slot)  # port outline
            # port prism at this level
            port_proto[parity, side] = make_prism_from_polygon_vertex(footprint, zb, zb + He_manifold_port_height)

    u_tools = {}      # level -> U fluid tool
    port_tools = {}   # (level, side) -> port fluid tool

    if He_network_boolean_mode == "sequential":  # alternative mode: cut every piece one at a time (slow, not used)
        stage(f"Copying U and ports to {N} levels")  # stage message
        u_tools[0] = u_proto  # level 0 uses the prototype
        for level in range(1, N):  # other levels
            u_tools[level] = copy_volume_move(u_proto, 0.0, 0.0, level * tube_pitch)  # copy the U up
        for level in range(N):  # every level
            parity = level % 2  # 0 or 1
            shift = (level - parity) * tube_pitch  # shift from the prototype level
            for side in ("left", "right"):  # both sides
                proto = port_proto[(parity, side)]  # matching prototype
                # prototype at its own level, otherwise a copy
                port_tools[level, side] = proto if shift == 0.0 else copy_volume_move(proto, 0.0, 0.0, shift)

        # -- Cut in the G9 order: slots, U channels, ports ----------------------
        stage("Cutting 6 manifold slots")
        for key, tool in manifold_tools.items():  # each slot
            rafm = subtract_and_check(rafm, tool, f"{key[0]} slot {key[1]+1}")  # cut it

        stage(f"Cutting {N} U channels")  # stage message
        for level in range(N):  # each level
            rafm = subtract_and_check(rafm, u_tools[level], f"U level {level+1}")  # cut its U

        stage(f"Cutting {2*N} ports")  # stage message
        for level in range(N):  # each level
            for side in ("left", "right"):  # both sides
                # cut the port
                rafm = subtract_and_check(rafm, port_tools[level, side], f'port level {level + 1} {side}')

        stage("Uniting retained He tools into ODD / EVEN circuits")  # stage message
        # ODD circuit slots
        odd_ids = [manifold_tools['left', MANIFOLD_SLOT_MIDDLE], manifold_tools['right', MANIFOLD_SLOT_REAR]]
        # EVEN circuit slots
        even_ids = [manifold_tools['left', MANIFOLD_SLOT_REAR], manifold_tools['right', MANIFOLD_SLOT_MIDDLE]]
        for level in range(N):  # each level
            group = odd_ids if level % 2 == 0 else even_ids  # its circuit
            # add the U and both ports
            group.extend([u_tools[level], port_tools[level, 'left'], port_tools[level, 'right']])
        odd = unite_retained_fluid_group(odd_ids, "IB_HE_ODD_LMIDDLE_RREAR")  # unite the ODD circuit
        # unite the EVEN circuit
        even = unite_retained_fluid_group(even_ids, 'IB_HE_EVEN_LREAR_RMIDDLE') if N > 1 else None

    else:  # "united"
        # 2-level unit cell: cell 0 = U(level 0) + its 2 ports, cell 1 = U(level 1)
        # + its 2 ports. Each cell is copied up by 2*pitch.
        stage("Building 2-level unit cell")
        # BUG FIX (v7): copy the bare U for level 1 BEFORE cell 0 is united.
        # Cubit's unite keeps the first volume's ID, so after uniting cell 0 the
        # ID u_proto already holds U + the two level-0 ports. Copying it then
        # (as v4-v6 did) gave every EVEN level FOUR ports: its own two plus the
        # ODD pair, connecting each even channel into the ODD slots as well.
        u1 = copy_volume_move(u_proto, 0.0, 0.0, tube_pitch) if N > 1 else None  # bare U at level 1
        # cell 0 = U + its 2 ports, united
        cell = {0: unite_retained_fluid_group(
            [u_proto, port_proto[(0, "left")], port_proto[(0, "right")]],
            "IB_HE_CELL_ODD")}
        if N > 1:  # need a second level
            # cell 1 = U + its 2 ports, united
            cell[1] = unite_retained_fluid_group([u1, port_proto[1, 'left'], port_proto[1, 'right']], 'IB_HE_CELL_EVEN')
            # check: the two cells are mirror images (U + one 2 mm and one 0.5 mm
            # port each), so their volumes must agree; extra ports would not
            v_odd, v_even = cubit.volume(cell[0]).volume(), cubit.volume(cell[1]).volume()  # cell volumes
            if abs(v_even - v_odd) > 1.0e-3 * v_odd:  # more than 0.1 % apart
                raise RuntimeError(f"IB unit cells differ: ODD {v_odd*1e9:.0f} mm^3, EVEN {v_even*1e9:.0f} mm^3 "
                                   f"(a cell has the wrong number of ports)")  # stop before copying 323 levels
            log(f"  unit cells OK: ODD {v_odd*1e9:.0f} mm^3, EVEN {v_even*1e9:.0f} mm^3 (2 ports each)")  # report

        stage(f"Copying unit cell to {N} levels")  # stage message
        # ODD circuit starts with its 2 slots
        odd_ids = [manifold_tools['left', MANIFOLD_SLOT_MIDDLE], manifold_tools['right', MANIFOLD_SLOT_REAR]]
        # EVEN circuit starts with its 2 slots
        even_ids = [manifold_tools['left', MANIFOLD_SLOT_REAR], manifold_tools['right', MANIFOLD_SLOT_MIDDLE]]
        for level in range(N):  # every level
            parity = level % 2  # 0 or 1
            shift = (level - parity) * tube_pitch  # shift from the cell's level
            # cell at its own level, otherwise a Z copy
            vid = cell[parity] if shift == 0.0 else copy_volume_move(cell[parity], 0.0, 0.0, shift)
            (odd_ids if parity == 0 else even_ids).append(vid)  # add it to its circuit

        stage("Uniting ODD / EVEN circuits")  # stage message
        # ODD circuit = 2 slots + all even-index cells
        odd = unite_retained_fluid_group(odd_ids, "IB_HE_ODD_LMIDDLE_RREAR")
        # EVEN circuit = 2 slots + all odd-index cells
        even = unite_retained_fluid_group(even_ids, 'IB_HE_EVEN_LREAR_RMIDDLE') if N > 1 else None

        stage("Cutting 4 He network bodies from the RAFM")  # stage message
        # cut the unused left front slot
        rafm = subtract_and_check(rafm, manifold_tools[("left", MANIFOLD_SLOT_FRONT)], "left front slot")
        # cut the unused right front slot
        rafm = subtract_and_check(rafm, manifold_tools[("right", MANIFOLD_SLOT_FRONT)], "right front slot")
        rafm = subtract_and_check(rafm, odd, "ODD circuit")  # cut the whole ODD circuit at once
        if even is not None:  # normal case
            rafm = subtract_and_check(rafm, even, "EVEN circuit")  # cut the whole EVEN circuit at once
        else:
            for vid in even_ids:  # one level only: cut the two EVEN slots alone
                rafm = subtract_and_check(rafm, vid, f"unused slot {vid}")  # cut it

    left_front = manifold_tools[("left", MANIFOLD_SLOT_FRONT)]  # left front slot body
    right_front = manifold_tools[("right", MANIFOLD_SLOT_FRONT)]  # right front slot body
    register_volume(left_front, "IB_HE_LEFT_UNUSED_FRONT_MANIFOLD")  # name it
    register_volume(right_front, "IB_HE_RIGHT_UNUSED_FRONT_MANIFOLD")  # name it

    # IB He network bodies
    he_networks = [("IB_HE_LEFT_UNUSED_FRONT_MANIFOLD", left_front),
                   ("IB_HE_RIGHT_UNUSED_FRONT_MANIFOLD", right_front),
                   ("IB_HE_ODD_LMIDDLE_RREAR", odd)]
    if even is not None:  # normal case
        he_networks.append(("IB_HE_EVEN_LREAR_RMIDDLE", even))  # add the EVEN circuit
    else:
        # N == 1: the two even-circuit slots are already cut but connect to nothing.
        for vid in even_ids:
            register_volume(vid, f"IB_HE_UNUSED_SLOT_{vid}")  # name the unused slot
            he_networks.append((f"IB_HE_UNUSED_SLOT_{vid}", vid))  # add it

    yield "IB phase 1 done"      # body + wrapped-He network (cut into the plain body)

    # ---------------------------------------------------------------------------
    # PHASE 2 (B). PbLi ducts + SiC inserts (full height)
    # ---------------------------------------------------------------------------
    stage("PbLi ducts (10 retained fluid cutters)")
    pbli_layout = get_PbLi_layout()  # 10 duct outlines
    rafm, pbli_fluid_ids = cut_PbLi_ducts(rafm, pbli_layout)  # cut them (kept as PbLi)

    stage("SiC inserts (10 SiC shells, 20 PbLi bodies)")  # stage message
    # split into PbLi gap / SiC / PbLi core
    pbli_fluid_ids, sic_solid_ids = build_SiC_inserts_and_split_PbLi(pbli_layout, pbli_fluid_ids)

    # ---------------------------------------------------------------------------
    # PHASE 2 (C). Internal He holes: prototype + rotate/translate copies (full height)
    # ---------------------------------------------------------------------------
    stage("Internal He holes (prototype + copies, batched subtract)")
    # cut all internal He holes (kept as He)
    rafm, he_hole_count, he_hole_polygons, internal_He_fluid_ids = cut_patterned_he_holes(
        rafm, body_polygon, pbli_layout
    )
    for i, vid in enumerate(internal_He_fluid_ids, 1):  # each hole body
        register_volume(vid, f"IB_INTERNAL_HE_{i:04d}")  # name it
    print(f"  Retained {len(internal_He_fluid_ids)} internal-He fluid bodies")  # report the count

    # ---------------------------------------------------------------------------
    # PHASE 2 (D). Optional interface check against the final RAFM (off by default)
    # ---------------------------------------------------------------------------
    if Trim_He_networks_against_RAFM:
        stage("Interface check: trimming He networks against final RAFM")  # stage message
        # trim each network
        he_networks = [(name, conform_fluid_to_final_rafm(vid, rafm, name)) for name, vid in he_networks]

    # ---------------------------------------------------------------------------
    # PHASE 2 (E). Bookkeeping (blocks, save and export are done by the DRIVER)
    # ---------------------------------------------------------------------------
    stage("Final checks")
    register_volume(rafm, f"IB_RAFM_{N}CH")  # name the RAFM
    he_network_ids = [vid for _, vid in he_networks]  # He network IDs
    rafm_ids = [rafm]  # one RAFM body
    helium_ids = he_network_ids + list(internal_He_fluid_ids)  # all inboard He bodies
    pbli_ids = list(pbli_fluid_ids)  # all inboard PbLi bodies
    sic_ids = list(sic_solid_ids)  # all inboard SiC bodies
    material_ids = rafm_ids + helium_ids + pbli_ids + sic_ids  # every inboard volume
    expected_ids = set(material_ids)  # as a set
    if len(material_ids) != len(expected_ids):  # no ID may appear twice
        raise RuntimeError("IB: duplicate IDs assigned to material families")  # stop with a clear message
    missing = expected_ids - set(cubit.get_entities("volume"))  # IDs that no longer exist
    if missing:  # any missing
        raise RuntimeError(f"IB: volumes missing at the end: {sorted(missing)}")  # stop with a clear message
    for vid in expected_ids:  # every volume
        assert_positive_solid(vid, "final")  # must be a real solid

    # count summary
    print(f"  RAFM: 1 | He networks: {len(he_network_ids)} | internal He: "
          f"{len(internal_He_fluid_ids)} | PbLi: {len(pbli_ids)} | SiC: {len(sic_ids)}")
    print(f"  TOTAL volumes: {len(expected_ids)}")  # total

    # phase 2 done: hand the material lists to the driver
    yield {"rafm": rafm_ids, "he": helium_ids, "pbli": pbli_ids, "sic": sic_ids}


def build_outboard():  # whole outboard script as a generator: geometry info, phase 1, phase 2
    # ============================================================================
    # FNSF_DCLL_OB_wrapped.py
    #
    # FNSF dual-coolant lead-lithium (DCLL) OUTBOARD blanket sector, built in
    # Coreform Cubit from user inputs. Writes one .cub and one .stp.
    #
    # Run from the Cubit Python console / script tab:
    #   exec(open(r"C:\Users\jessi\Downloads\FNSF_DCLL_OB_wrapped.py").read())
    #
    # UNITS: METERS (same as the inboard script). Machine axis = Z, the sector
    # spans theta = 0 .. 360/Num_sectors degrees from +X (same as the inboard).
    #
    # GEOMETRY
    #   The plasma boundary is the FNSF shaped curve
    #       R = R0 + a cos(xi + tau sin xi),   Z = kappa a sin xi
    #   Every outboard surface is a NORMAL OFFSET ("depth") of that curve. The
    #   plasma-facing FW surface sits at depth tSOL. The outboard runs poloidally
    #   between the points where the FW surface reaches R = R0 (top and bottom).
    #
    #   Radial build (from the FW surface inward, along the local normal):
    #     FW / wrapped-He band (38 mm, wrapped-He U channels cut into it)
    #     front gap
    #     PbLi row 1 | radial web (large He holes) | row 2 | web | row 3 | web
    #     | row 4 | rear He-row structure | extra back wall (+10 mm)
    #
    #   Toroidal build (each PbLi row): 9 ducts between the two 38 mm side He
    #   bands. Ducts 2-8 have equal width; ducts 1 and 9 = ratio x center width
    #   (ratio taken from the inboard). Columns are constant-angle wedges sized
    #   at the row's NARROWEST radius, so every web is >= its nominal width and
    #   the sector expansion toward the midplane goes into the ducts and webs.
    #   Edge-duct outer sides are parallel to the sector walls.
    #
    #   All PbLi ducts and internal He holes run POLOIDALLY along the curve and
    #   are OPEN at the top and bottom: they end flush with the RAFM end faces
    #   (the end faces are the plasma-normal lines at the two end points, revolved).
    #
    # FLUIDS RETAINED (every cutter is kept with keep_tool)
    #   PbLi: outer gap + core per duct; SiC: one shell per duct;
    #   He: every internal He hole and every wrapped-He channel.
    #   Blocks MAT_RAFM / MAT_HE / MAT_PBLI / MAT_SIC.
    #
    # WRAPPED-He CHANNELS (inboard profile: 30 x 16.12 mm void, 20.12 mm pitch)
    #   U channels inside the 38 mm band: across the FW, around the rounded FW
    #   corners, straight back through each side band along the LOCAL plasma
    #   normal and OPEN at the back face. Same void size everywhere; the RAFM
    #   between channels grows toward the back as the normals fan out.
    #
    # SPEED
    #   - Duct corner fillets (R50.8) and FW corner rounding (R88.8) are NOT ACIS
    #     blends. Each is made by subtracting a thin swept "corner trimmer"
    #     (square minus quarter circle) from a simple revolved body BEFORE that
    #     body touches the RAFM.
    #   - Internal He holes are built EXACTLY like the ducts: a revolved band with
    #     its R3 corners made by the same swept corner trimmers, constant-angle
    #     (they widen with the sector like the ducts), open at the top/bottom.
    #   - Anything repeated toroidally is built ONCE and rotate-copied about Z
    #     (center ducts 3-8 and their SiC, every large-He row, the small He holes
    #     in webs 2-8).
    #   - Order in this combined file: wrapped He first (plain body) -> PbLi
    #     ducts + SiC -> internal He holes.
    # ============================================================================


    # ============================================================================
    # USER INPUTS
    # ============================================================================

    # ---------------------------------------------------------------------------
    # Machine / plasma (FNSF)
    # ---------------------------------------------------------------------------

    # Poloidal extent: the outboard ends where the FW surface reaches this radius.


    # ---------------------------------------------------------------------------
    # USER INPUTS -> outboard names. The outboard uses the SAME dimensions as
    # the inboard (USER INPUTS at the top of the file); it has no manifold.
    # ---------------------------------------------------------------------------
    He_band_thickness = 2.0 * He_channel_wall + He_channel_width  # FW band and the two side bands [m] (38 mm)
    FW_corner_outer_radius = FW_corner_radius  # rounded FW / side-wall corner [m]
    Num_PbLi_rows = int(Num_PbLi_rows_OB)  # PbLi rows (columns are calculated per row)
    front_gap_after_He = 0.0  # PbLi row 1 starts at the band's inner face
    PbLi_edge_gap_from_wrapped_He = 0.0  # edge ducts start at the band's inner face
    PbLi_intercolumn_web_width = PbLi_column_gap  # web between duct columns (at the narrowest radius) [m]
    PbLi_row_radial_depths = [PbLi_row1_length] + [PbLi_other_rows_length] * (Num_PbLi_rows - 1)  # row depths [m]
    PbLi_corner_radius = PbLi_duct_fillet_radius  # duct corner radius [m]
    OB_extra_back_wall_thickness = 0.0  # optional extra RAFM behind the rear holes [m] (0 = same as the inboard)
    SiC_offset_from_PbLi_cavity = SiC_gap  # PbLi gap on the sides and back [m]
    SiC_wall_thickness = SiC_thickness  # SiC wall [m]
    SiC_front_gap_by_row = [SiC_front_gap_row1] + [SiC_gap] * (Num_PbLi_rows - 1)  # PbLi gap at each row's front [m]
    web_He_hole_across_web = Divider_He_hole_length  # row-divider hole radial size [m]
    web_He_hole_along_arc = Divider_He_hole_width  # row-divider hole width [m]
    web_He_hole_corner_radius = Divider_He_hole_fillet_radius  # corner radius [m]
    web_He_min_interhole_rib = Divider_He_hole_rib  # fixed rib between holes [m]
    web_He_hole_RAFM_ligament = 0.5 * (PbLi_row_gap - Divider_He_hole_length)  # RAFM each side of the holes [m]
    rear_He_hole_across_web = Rear_He_hole_length  # rear hole radial size [m]
    rear_He_hole_along_arc = Rear_He_hole_width  # rear hole width [m]
    rear_He_hole_corner_radius = Divider_He_hole_fillet_radius  # corner radius [m]
    rear_He_min_interhole_rib = Rear_He_hole_rib  # fixed rib between holes [m]
    small_He_hole_radial_length = Column_He_hole_length  # column hole length along the web [m]
    small_He_hole_toroidal_width = Column_He_hole_width  # column hole width across the web [m]
    small_He_hole_corner_radius = Column_He_hole_fillet_radius  # corner radius [m]
    small_He_interhole_gap = Column_He_hole_rib  # fixed rib between column holes [m]
    He_void_width = He_channel_width  # wrapped-He void across the 38 mm band [m]
    He_void_height = He_channel_height  # wrapped-He void height [m]
    He_top_wall = 0.5 * He_channel_stack_wall  # (only used for the IB-style pitch reference)
    He_bottom_wall = 0.5 * He_channel_stack_wall

    # He-hole corner fillets (the corner radii above).
    #   False = SHARP-cornered He holes: fast (16 holes cut in ~0.6 s in testing).
    #   True  = true R3 fillets: ~80x slower to cut (~50 s per 16 holes, and
    #           each batch gets slower), so use it for the FINAL build only.
    # PbLi duct fillets (R50.8) are always built.
    He_hole_fillets = False

    # ---------------------------------------------------------------------------
    # Build switches
    # ---------------------------------------------------------------------------
    Build_SiC_inserts = True
    Build_web_He_holes = True  # True = large He holes in the radial webs
    Build_rear_He_holes = True  # True = rear He row
    Build_small_He_holes = True  # True = small He holes in the column webs
    Build_wrapped_He_channels = True  # True = wrapped-He U channels in the FW / side band

    # ---------------------------------------------------------------------------
    # Resolution / performance
    # ---------------------------------------------------------------------------
    # Spline points along the 5.6-7.4 m poloidal curves. 60 points follow the
    # FNSF curve to ~2 um (180 -> 0.03 um but makes every Boolean much slower).
    Body_profile_segments = 60       # revolved RZ profiles (body, ducts, He holes)
    Sweep_path_points = 60           # corner-trimmer sweep paths
    Arc_length_samples = 8000
    Corner_polyline_points = 6       # only used if Cubit can't make 3-point arcs
    Trimmer_overrun = 0.05           # corner trimmers run past duct ends [m]
    Trimmer_outer_margin = 0.002     # trimmer extends this far outside the faces
    Trimmer_arc_push = 0.00005       # pushes trimmer arc ends just outside the faces
    # PbLi/SiC ducts: corner-trimmer arc push for the first build, then for an
    # automatic rebuild if a duct fails ACIS validate.
    Duct_trimmer_push_retries = [0.00005, 0.001, 0.002]
    # Per-duct override {(row, column) 1-based: [push, ...]}. R1 C2 (copied to
    # C3..C8) is the one duct that is unreliable at 50 um: tested in Cubit AND
    # SolidWorks, only 1.0 mm gave valid ACIS + 3 solid bodies (50 um: ACIS
    # error; 200 / 500 um: surface bodies in SolidWorks). Fillet deviation
    # from a perfect R50.8 is <= 1 mm. All other ducts imported clean at 50 um.
    Duct_trimmer_push_override = {(1, 2): [0.001, 0.002]}
    Boolean_batch_size = 16  # cutters per Boolean

    # ---------------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------------

    # Graphics on/off and the output names are SHARED INPUTS at the top of the
    # file (Show_build_in_graphics, Output_directory, Output_basename).

    # ============================================================================
    # END OF USER INPUTS
    # ============================================================================


    Sector_angle = 360.0 / Num_sectors  # sector angle [deg]
    b = a * kappa  # plasma half-height b = kappa * a [m]
    EZ = np.array([0.0, 0.0, 1.0])  # unit vector along the machine axis


    # ============================================================================
    # FNSF CURVE AND ITS NORMAL OFFSETS
    # ============================================================================

    def XZcoords(xi, depth):  # vectorized (R, Z) on the offset curve at angle(s) xi and a depth
        """(R, Z) of the curve at normal depth `depth` from the plasma boundary."""
        x = np.radians(np.asarray(xi, dtype=float))
        base = np.array([R0 + a * np.cos(x + tau * np.sin(x)), b * np.sin(x)])  # plasma boundary point(s)
        # outward normal(s), not yet normalized
        nrm = np.array([b * np.cos(x), a * np.sin(x + tau * np.sin(x)) * (1.0 + tau * np.cos(x))])
        return base + nrm * (depth / np.linalg.norm(nrm, axis=0))  # boundary + depth along the unit normal


    def rz(xi, depth):  # scalar (R, Z) at one xi and depth
        p = XZcoords(float(xi), depth)  # vector version
        return float(p[0]), float(p[1])  # plain floats


    def rz_normal(xi):  # unit normal of the curve at xi
        """Unit RZ normal pointing to increasing depth (away from plasma)."""
        x = math.radians(float(xi))
        # normal components
        n = np.array([b * math.cos(x), a * math.sin(x + tau * math.sin(x)) * (1.0 + tau * math.cos(x))])
        return n / np.linalg.norm(n)  # normalized


    class ArcMap:  # lookup table between poloidal angle xi and arc length at a fixed depth
        """Arc length <-> xi along the curve at one depth."""  # docstring

        def __init__(self, depth, xi_a, xi_b, n=None):  # build the table
            n = n or Arc_length_samples  # default sample count
            self.depth = depth  # depth of this curve
            self.xi = np.linspace(xi_a, xi_b, n + 1)  # sample angles
            p = XZcoords(self.xi, depth)  # sample points
            ds = np.hypot(np.diff(p[0]), np.diff(p[1]))  # segment lengths
            self.s = np.concatenate([[0.0], np.cumsum(ds)])  # cumulative arc length

        def s_at(self, xi):  # arc length at xi
            return float(np.interp(xi, self.xi, self.s))  # linear interpolation

        def xi_at(self, s):  # xi at arc length s
            if s < -1e-9 or s > self.s[-1] + 1e-9:  # must lie on the mapped stretch
                raise ValueError("Arc-length location is outside the mapped curve.")  # stop with a clear message
            return float(np.interp(s, self.s, self.xi))  # linear interpolation

        def equal_arc_xi(self, xi_a, xi_b, n_seg):  # n_seg + 1 angles equally spaced in arc length
            sa, sb = self.s_at(xi_a), self.s_at(xi_b)  # arc length at both ends
            return [self.xi_at(s) for s in np.linspace(sa, sb, n_seg + 1)]  # equal arc spacing


    def find_xi_for_radius(target_R, depth):  # xi (lower half) where the curve at this depth reaches radius target_R
        lo, hi = -89.9, 0.0  # bracket below the midplane
        f = lambda x: rz(x, depth)[0] - target_R  # signed distance from the target radius
        if f(lo) * f(hi) > 0.0:  # bracket must straddle the target
            raise ValueError("OB_end_radius is not reached on the FW surface.")  # stop with a clear message
        for _ in range(200):  # bisection
            mid = 0.5 * (lo + hi)  # midpoint
            if f(lo) * f(mid) <= 0.0:  # root in the lower half
                hi = mid  # move the upper bound
            else:
                lo = mid  # move the lower bound
        return 0.5 * (lo + hi)  # root xi [deg]


    # Map range with headroom for trimmer overrun
    _XI_EXT = 8.0


    def arcmap(depth):  # arc map at this depth covering the blanket plus headroom
        return ArcMap(depth, OB_xi_min_full - _XI_EXT, OB_xi_max_full + _XI_EXT)  # map range


    def feature_xi_range(depth):  # xi range of every PbLi duct and He hole (= the full blanket)
        """Poloidal xi range of a feature: the same end points as the RAFM body,
        so every duct and He hole is open at the top and bottom."""
        return OB_xi_min, OB_xi_max


    def extended_xi_range(depth, xi_a, xi_b, overrun):  # xi range widened by an arc length 'overrun' at both ends
        m = arcmap(depth)  # arc map at this depth
        return m.xi_at(m.s_at(xi_a) - overrun), m.xi_at(m.s_at(xi_b) + overrun)  # shift both ends along the curve


    def min_R_over(depths, xi_a, xi_b):  # smallest R reached by the curves at these depths over the xi range
        xs = np.linspace(xi_a, xi_b, 4001)  # sample angles
        return min(float(XZcoords(xs, d)[0].min()) for d in depths)  # minimum over all depths


    def max_R_over(depths, xi_a, xi_b):  # largest R reached by the curves at these depths over the xi range
        xs = np.linspace(xi_a, xi_b, 4001)  # sample angles
        return max(float(XZcoords(xs, d)[0].max()) for d in depths)  # maximum over all depths


    # ============================================================================
    # DERIVED GLOBAL GEOMETRY
    # ============================================================================

    OB_xi_max_full = -find_xi_for_radius(OB_end_radius, tSOL)  # top end: FW reaches R = OB_end_radius (symmetric)
    OB_xi_min_full = -OB_xi_max_full  # bottom end
    OB_xi_min, OB_xi_max = OB_xi_min_full, OB_xi_max_full   # whole outboard, always

    FW_depth = tSOL  # FW surface depth from the plasma boundary
    Side_offset = He_band_thickness + PbLi_edge_gap_from_wrapped_He  # wall -> ducts 1/9

    # Radial build, FW surface -> back face (everything here is an input):
    #   He band + front gap + rows + (rows-1) x radial webs + rear structure
    #   + rear manifold zone
    radial_web_thickness = 2.0 * web_He_hole_RAFM_ligament + web_He_hole_across_web
    # rear structure = clearance + hole + clearance
    rear_structure_thickness = Rear_He_gap_to_ducts + rear_He_hole_across_web + Rear_He_gap_to_back_wall
    if len(PbLi_row_radial_depths) != Num_PbLi_rows:  # one depth per row
        raise ValueError("PbLi_row_radial_depths needs one value per PbLi row.")  # stop with a clear message
    if any(h <= 0.0 for h in PbLi_row_radial_depths):  # all row depths positive
        raise ValueError("Every PbLi row depth must be > 0.")  # stop with a clear message
    PbLi_row_depths = [float(h) for h in PbLi_row_radial_depths]  # as floats
    # total OB depth, FW surface -> back face [m]
    total_OB_blanket_depth = (He_band_thickness + front_gap_after_He + sum(PbLi_row_depths)
                              + (Num_PbLi_rows - 1) * radial_web_thickness
                              + rear_structure_thickness + OB_extra_back_wall_thickness)
    Back_depth = tSOL + total_OB_blanket_depth  # back face depth from the plasma boundary

    # Hand the OB end geometry to the driver BEFORE anything is built, so the
    # inboard is sized to the FULL outboard height (its outer/back curve).
    yield {"back_corner_Z": float(XZcoords(OB_xi_max, Back_depth)[1]),
           "fw_end_Z": float(XZcoords(OB_xi_max, FW_depth)[1])}

    # Wrapped-He derived geometry
    He_wall = 0.5 * (He_band_thickness - He_void_width)          # 4 mm front/back/side walls
    He_pitch = He_bottom_wall + He_void_height + He_top_wall     # 20.12 mm
    He_channel_center_depth = FW_depth + 0.5 * He_band_thickness
    He_void_outer_bend_radius = FW_corner_outer_radius - He_wall                 # 84.8 mm
    He_void_inner_bend_radius = He_void_outer_bend_radius - He_void_width        # 54.8 mm

    # Row depth bands (d0 = front, d1 = back)
    PbLi_rows = []
    _d = FW_depth + He_band_thickness + front_gap_after_He  # row 1 starts behind the He band + front gap
    for _i, _h in enumerate(PbLi_row_depths):  # each row
        PbLi_rows.append({"row": _i, "d0": _d, "d1": _d + _h})  # its front and back depth
        _d += _h + (radial_web_thickness if _i < Num_PbLi_rows - 1 else 0.0)  # next row starts after this row + a web

    # Large-hole rows: one in each radial web, then the rear row
    He_hole_rows = []
    for _i in range(Num_PbLi_rows - 1):  # one large-He row in each web between PbLi rows
        # web row: center depth halfway between the rows
        He_hole_rows.append({
            "name": f"WEB{_i+1}",
            "dc": 0.5 * (PbLi_rows[_i]["d1"] + PbLi_rows[_i + 1]["d0"]),
            "across": web_He_hole_across_web, "along": web_He_hole_along_arc,
            "r": web_He_hole_corner_radius, "rib": web_He_min_interhole_rib,
            "build": Build_web_He_holes})
    # rear row behind the last PbLi row
    He_hole_rows.append({
        "name": "REAR",
        "dc": PbLi_rows[-1]["d1"] + Rear_He_gap_to_ducts + 0.5 * rear_He_hole_across_web,
        "across": rear_He_hole_across_web, "along": rear_He_hole_along_arc,
        "r": rear_He_hole_corner_radius, "rib": rear_He_min_interhole_rib,
        "build": Build_rear_He_holes})

    _layer_insets = [0.0]  # cavity layer (inset 0)
    if Build_SiC_inserts:  # with SiC: two more layers
        # SiC outer (2 mm) and SiC inner / PbLi core (7 mm) insets
        _layer_insets += [SiC_offset_from_PbLi_cavity, SiC_offset_from_PbLi_cavity + SiC_wall_thickness]


    def column_layout(row):  # toroidal layout of one PbLi row (constant-angle wedges, Num_PbLi_cols_OB columns)
        """Num_PbLi_cols_OB EQUAL wedges (same rule as the inboard): a web of
        PbLi_intercolumn_web_width (at the row's narrowest radius Rref) centred on
        every wedge line; the edge ducts run from the wrapped-He band to the first web."""
        xa, xb = feature_xi_range(0.5 * (row["d0"] + row["d1"]))
        Rref = min_R_over([row["d0"], row["d1"]], xa, xb)  # narrowest radius of the row
        Rmax = max_R_over([row["d0"], row["d1"]], xa, xb)  # widest radius of the row
        thL = math.degrees(math.asin(Side_offset / Rref))  # left usable edge angle at Rref
        thR = Sector_angle - thL  # right usable edge angle
        usable = Rref * math.radians(thR - thL)  # usable arc at Rref
        ncols = int(Num_PbLi_cols_OB)  # columns (input)
        if ncols < 3:
            raise ValueError("Num_PbLi_cols_OB must be at least 3.")
        w = PbLi_intercolumn_web_width  # web width
        deg = lambda L: math.degrees(L / Rref)  # arc length at Rref -> angle
        alpha = Sector_angle / ncols  # equal wedge angle [deg]
        hw = 0.5 * deg(w)  # half a web as an angle at Rref
        dividers = [k * alpha for k in range(1, ncols)]  # wedge lines [deg]
        edges = [thL] + [x for d in dividers for x in (d - hw, d + hw)] + [thR]  # duct boundaries
        cols = [(edges[2 * c], edges[2 * c + 1]) for c in range(ncols)]  # (start, end) angle of each column
        center = Rref * math.radians(cols[1][1] - cols[1][0])  # inner duct width at Rref
        edge = Rref * math.radians(cols[0][1] - cols[0][0])  # edge duct width at Rref
        if min(center, edge) <= 2.0 * PbLi_corner_radius:  # a duct must hold its two corner fillets
            raise ValueError(f"OB row {row['row']+1}: {ncols} columns leave a duct only "
                             f"{min(center, edge)*1000:.1f} mm wide (less than two {PbLi_corner_radius*1000:.1f} mm fillets).")
        rho = edge / center  # resulting edge / inner ratio (for the log)
        webs = [0.5 * (cols[c][1] + cols[c + 1][0]) for c in range(ncols - 1)]  # web center angles
        return dict(xi=(xa, xb), Rref=Rref, Rmax=Rmax, cols=cols, webs=webs, ncols=ncols, ratio=rho,
                    center=center, edge=edge, pitch=cols[2][0] - cols[1][0] if ncols > 3 else 0.0,
                    web_angle=deg(w))


    def fixed_rib_centered(L, w, rib, end_min, label=None):  # most holes with a FIXED rib, leftover split at both ends
        """Holes of size w with exactly `rib` between them, as many as fit in L with at
        least end_min left at both ends; centred. Returns (centres, end)."""
        n = int(math.floor((L + rib) / (w + rib) + 1.0e-9))  # how many fit
        while n > 0 and 0.5 * (L - n * w - (n - 1) * rib) < end_min - 1.0e-12:
            n -= 1  # keep the end ligaments
        if n < 1:  # nothing fits
            if label:
                raise ValueError(f"{label}: no hole fits.")
            return [], None
        end = 0.5 * (L - n * w - (n - 1) * rib)  # leftover at each end
        return [end + 0.5 * w + i * (w + rib) for i in range(n)], end  # (centres, end)


    # ============================================================================
    # CUBIT HELPERS
    # ============================================================================

    def _last(kind):  # last-created entity ID of a type
        return cubit.get_last_id(kind)  # Cubit lookup


    def new_vertex(p):  # create a vertex at a 3D point and return its ID
        cubit.cmd(f"create vertex {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}")  # Cubit command
        return _last("vertex")  # its ID


    def new_volumes_since(before):  # volume IDs created since 'before'
        return list(set(cubit.get_entities("volume")) - before)  # set difference


    def run_sweep(command, label):  # run a sweep command and return the solid it made
        """Run a sweep and return the solid it made. Cubit stores a free surface
        as a SHEET body that already has a volume ID; sweeping usually turns that
        same volume into the solid, so often no NEW volume ID appears."""
        before = set(cubit.get_entities("volume"))
        last_before = cubit.get_last_id("volume")  # highest volume ID before the sweep
        cubit.cmd(command)  # Cubit: the sweep
        new = new_volumes_since(before)  # IDs created
        if len(new) == 1:  # one new volume
            vid = new[0]  # that is the solid
        # no new ID: the sheet body itself became the solid
        elif not new and last_before in set(cubit.get_entities("volume")):
            vid = last_before  # same ID
        else:
            raise RuntimeError(f"{label}: sweep produced volumes {sorted(new)}")  # stop with a clear message
        v = cubit.volume(vid)  # Cubit volume object
        if v.volume() <= 0 or any(bd.is_sheet_body() for bd in v.bodies()):  # must be a real solid
            raise RuntimeError(f"{label}: sweep did not produce a solid (volume {vid})")  # stop with a clear message
        return vid  # solid ID


    def arcs_available():  # check (once) whether Cubit can make 3-point arcs
        """Probe once whether 'create curve arc three vertex' works."""
        global _ARC_OK
        if _ARC_OK is None:  # not probed yet
            v = [new_vertex(p) for p in ((0, 0, 0), (1, 1, 0), (2, 0, 0))]  # three test vertices
            c0 = _last("curve")  # last curve ID before
            try:  # the command may fail
                cubit.cmd(f"create curve arc three vertex {v[0]} {v[1]} {v[2]}")  # try an arc
            except Exception:
                pass  # failed: arcs unavailable
            c1 = _last("curve")  # last curve ID after
            _ARC_OK = c1 > c0  # a new curve = arcs work
            if _ARC_OK:  # clean up the test arc
                cubit.cmd(f"delete curve {c1}")          # also removes its 2 end vertices
                cubit.cmd(f"delete vertex {v[1]}")       # the free middle vertex
            else:
                cubit.cmd("delete vertex " + " ".join(map(str, v)))  # clean up the test vertices
            print(f"  3-point arcs available: {_ARC_OK}")  # report the result
        return _ARC_OK  # True / False


    def make_profile_surface(segments):  # planar surface from line / arc segments
        """Planar surface from ordered segments in 3D:
           ('line', p0, p1) or ('arc', p0, pmid, p1). Consecutive segments share
           endpoints. Arcs fall back to polylines if 3-point arcs are unavailable."""
        use_arcs = arcs_available()  # probe arc support
        vcache = {}  # point -> vertex ID (shared corners)
        free_mids = []  # arc middle vertices (deleted afterwards)

        def vid(p):  # vertex ID for a point, created once
            key = tuple(round(float(x), 12) for x in p)  # rounded key
            if key not in vcache:  # new point
                vcache[key] = new_vertex(p)  # create it
            return vcache[key]  # vertex ID

        curves = []  # curve IDs around the outline
        for seg in segments:  # each segment
            if seg[0] == "line":  # straight segment
                cubit.cmd(f"create curve vertex {vid(seg[1])} {vid(seg[2])}")  # Cubit: straight curve
                curves.append(_last("curve"))  # record it
            elif use_arcs:  # arc segment
                vm = new_vertex(seg[2])  # middle vertex
                free_mids.append(vm)  # delete it later
                cubit.cmd(f"create curve arc three vertex {vid(seg[1])} {vm} {vid(seg[3])}")  # Cubit: 3-point arc
                curves.append(_last("curve"))  # record it
            else:
                p0, pm, p1 = (np.asarray(x, dtype=float) for x in seg[1:])  # arc fallback: polyline
                pts = arc_polyline(p0, pm, p1, Corner_polyline_points)  # polyline points on the arc
                for q0, q1 in zip(pts[:-1], pts[1:]):  # each small segment
                    cubit.cmd(f"create curve vertex {vid(q0)} {vid(q1)}")  # Cubit: straight curve
                    curves.append(_last("curve"))  # record it
        s0 = _last("surface")  # last surface ID before
        cubit.cmd("create surface curve " + " ".join(map(str, curves)))  # Cubit: surface from the closed curve loop
        s1 = _last("surface")  # last surface ID after
        if s1 == s0:  # no new surface
            raise RuntimeError("Profile surface creation failed.")  # stop with a clear message
        if free_mids:  # arc middle vertices exist
            cubit.cmd("delete vertex " + " ".join(map(str, free_mids)))  # delete them
        return s1  # surface ID


    def arc_polyline(p0, pm, p1, n):  # points on the circle through p0, pm, p1 (3D), from p0 to p1
        """Points on the circle through p0, pm, p1 from p0 to p1 (3D)."""
        a_, b_ = p0 - pm, p1 - pm
        axb = np.cross(a_, b_)  # plane normal (unnormalized)
        # circumcenter of the three points
        center = pm + (np.dot(a_, a_) * np.cross(b_, axb) + np.dot(b_, b_) * np.cross(axb, a_)) \
            / (2.0 * np.dot(axb, axb))
        e1 = (p0 - center)  # from center to p0
        r = np.linalg.norm(e1)  # radius
        e1 /= r  # unit vector to p0
        nrm = axb / np.linalg.norm(axb)  # unit plane normal
        e2 = np.cross(nrm, e1)  # second in-plane axis

        def ang(p):  # polar angle of a point in the (e1, e2) frame
            d = p - center  # from the center
            return math.atan2(np.dot(d, e2), np.dot(d, e1))  # angle

        am, a1 = ang(pm), ang(p1)  # angles of the middle and end points
        if am < 0:  # wrap into [0, 2 pi)
            am += 2 * math.pi  # wrap
        if a1 < 0:  # wrap into [0, 2 pi)
            a1 += 2 * math.pi  # wrap
        if a1 < am:     # go the other way
            a1 -= 2 * math.pi
        # n + 1 points from p0 to p1
        return [center + r * (math.cos(t) * e1 + math.sin(t) * e2) for t in np.linspace(0.0, a1, n + 1)]


    def make_spline_path(points):  # spline curve through 3D points
        """Spline curve through 3D points; tries 'location' form first so no
        construction vertices are left behind."""
        c0 = _last("curve")
        locs = " ".join(f"location {p[0]:.12g} {p[1]:.12g} {p[2]:.12g}" for p in points)  # "location x y z" list
        try:  # the location form may be unsupported
            cubit.cmd("create curve spline " + locs)  # Cubit: spline through locations
        except Exception:
            pass  # fall back below
        if _last("curve") > c0:  # it worked
            return _last("curve"), []  # curve ID, no vertices to clean up
        vids = [new_vertex(p) for p in points]  # fallback: make vertices first
        cubit.cmd("create curve spline vertex " + " ".join(map(str, vids)))  # Cubit: spline through vertices
        if _last("curve") == c0:  # still nothing
            raise RuntimeError("Spline path creation failed.")  # stop with a clear message
        return _last("curve"), vids  # curve ID + its vertices


    def sweep_along(surface_id, path_id, path_vids):  # sweep a surface along a path curve and clean the path up
        # Cubit: sweep along the curve
        vid = run_sweep(f'sweep surface {surface_id} along curve {path_id}', f'sweep along curve {path_id}')
        cubit.cmd(f"delete curve {path_id}")  # delete the path
        if len(path_vids) > 2:                         # ends go with the curve
            cubit.cmd("delete vertex " + " ".join(map(str, path_vids[1:-1])))
        return vid  # swept solid ID


    # planar R-Z patch (in y = 0) between two depths, bounded by splines
    def make_rz_patch_surface(d_in, d_out, xi_a, xi_b, n_seg):
        """Planar RZ patch (in the y=0 plane) between two depths. Same proven
        vertex/spline construction as the earlier outboard scripts."""
        def verts(depth):
            xs = arcmap(depth).equal_arc_xi(xi_a, xi_b, n_seg)  # equal-arc angles at this depth
            return [new_vertex((rz(x, depth)[0], 0.0, rz(x, depth)[1])) for x in xs]  # vertices on that curve
        vin, vout = verts(d_in), verts(d_out)  # inner and outer vertex rows
        curves = []  # four boundary curves
        cubit.cmd("create curve spline vertex " + " ".join(map(str, vin)))  # inner spline
        curves.append(_last("curve"))  # record it
        cubit.cmd(f"create curve vertex {vin[-1]} {vout[-1]}")  # end line at the top
        curves.append(_last("curve"))  # record it
        # outer spline, reversed to close the loop
        cubit.cmd("create curve spline vertex " + " ".join(map(str, reversed(vout))))
        curves.append(_last("curve"))  # record it
        cubit.cmd(f"create curve vertex {vout[0]} {vin[0]}")  # end line at the bottom
        curves.append(_last("curve"))  # record it
        s0 = _last("surface")  # last surface ID before
        cubit.cmd("create surface curve " + " ".join(map(str, curves)))  # Cubit: surface from the 4 curves
        if _last("surface") == s0:  # no new surface
            raise RuntimeError("RZ patch surface creation failed.")  # stop with a clear message
        return _last("surface")  # surface ID


    # revolve an R-Z patch about Z from th_start through th_sweep degrees
    def revolve_rz_patch(d_in, d_out, xi_a, xi_b, th_start, th_sweep, n_seg=None):
        n_seg = n_seg or Body_profile_segments  # default resolution
        s = make_rz_patch_surface(d_in, d_out, xi_a, xi_b, n_seg)  # the patch
        if abs(th_start) > 1e-12:  # start angle not zero
            cubit.cmd(f"rotate surface {s} angle {th_start:.12g} about z")  # Cubit: rotate the patch to the start angle
        return run_sweep(f"sweep surface {s} zaxis angle {th_sweep:.12g}", "revolve")  # Cubit: revolve about the Z axis


    def subtract(body, tools, keep_tool=False, label=""):  # subtract tools from a body; strict check of the result
        """Subtract tools from body; returns the (single) resulting body id."""
        tools = list(tools)
        if not tools:  # nothing to cut
            return body  # body unchanged
        before = set(cubit.get_entities("volume"))  # volumes before
        # Cubit: subtract, imprint, optionally keep the tools
        cubit.cmd(f"subtract volume {' '.join(map(str, tools))} from volume {body} imprint"
                  + (" keep_tool" if keep_tool else ""))
        after = set(cubit.get_entities("volume"))  # volumes after
        if keep_tool and not set(tools) <= after:  # kept tools must still exist
            # stop with a clear message
            raise RuntimeError(f"{label}: keep_tool lost cutters {sorted(set(tools) - after)}")
        made = after - before  # IDs created
        if body in after:  # body kept its ID
            if made:  # but the cut split it
                # stop with a clear message
                raise RuntimeError(f"{label}: body was split into extra volumes {sorted(made)}")
            return body  # same ID
        if len(made) != 1:  # must be exactly one new body
            # stop with a clear message
            raise RuntimeError(f'{label}: expected one result body, got {sorted(made)} (the body may have been split)')
        return next(iter(made))  # its ID


    def subtract_batched(body, tools, label):  # subtract many cutters in batches, keeping them (fluids)
        n = int(math.ceil(len(tools) / float(Boolean_batch_size)))  # number of batches
        for i in range(n):  # each batch
            batch = tools[i * Boolean_batch_size:(i + 1) * Boolean_batch_size]  # cutter IDs in this batch
            body = subtract(body, batch, keep_tool=True, label=label)  # cut them
            log(f"  {label}: batch {i+1}/{n} ({len(batch)} cutters)")  # progress line
            show()  # redraw if graphics are on
        return body  # final body ID


    def copy_rotate_z(vid, angle):  # rotate-copy a volume about Z (exact for this axisymmetric layout)
        before = set(cubit.get_entities("volume"))  # volumes before
        cubit.cmd(f"volume {vid} copy rotate {angle:.12g} about z")  # Cubit: copy + rotate
        new = new_volumes_since(before)  # IDs created
        if len(new) != 1:  # must be exactly one
            raise RuntimeError(f"Rotate-copy of volume {vid} failed.")  # stop with a clear message
        return new[0]  # copy ID


    def name_volume(vid, name):  # give a volume a readable name
        cubit.cmd(f"volume {vid} name '{name}'")  # Cubit command


    def assert_solid(vid, label):  # raise if a volume is empty or a sheet body
        v = cubit.volume(vid)  # Cubit volume object
        if v.volume() <= 0 or any(bd.is_sheet_body() for bd in v.bodies()):  # zero volume or a sheet body
            raise RuntimeError(f"{label}: volume {vid} is not a positive solid")  # stop with a clear message


    # ============================================================================
    # VERTICAL PLANES, PATHS AND PROFILES
    # ============================================================================

    def e_R(theta_deg):  # horizontal unit vector at angle theta
        t = math.radians(theta_deg)  # degrees -> radians
        return np.array([math.cos(t), math.sin(t), 0.0])  # (cos, sin, 0)


    def plane_left(theta_deg, c=0.0):  # plane on the LEFT side of a feature (interior toward +theta)
        """Vertical plane through the Z axis at theta, offset c, interior toward +theta."""
        t = math.radians(theta_deg)
        # n = unit normal, c = offset, m = in-plane radial direction
        return dict(n=np.array([-math.sin(t), math.cos(t), 0.0]), c=c, m=e_R(theta_deg), theta=theta_deg)


    def plane_right(theta_deg, c=0.0):  # plane on the RIGHT side of a feature (interior toward -theta)
        """Vertical plane at theta, offset c, interior toward -theta."""
        t = math.radians(theta_deg)
        # n = unit normal, c = offset, m = in-plane radial direction
        return dict(n=np.array([math.sin(t), -math.cos(t), 0.0]), c=c, m=e_R(theta_deg), theta=theta_deg)


    def offset_plane(pl, extra):  # copy of a plane moved 'extra' further along its normal
        q = dict(pl)  # copy
        q["c"] = pl["c"] + extra  # bigger offset
        return q  # new plane


    def point_on_plane(pl, xi, depth):  # 3D point at (xi, depth) on the offset curve, lying in vertical plane pl
        """Point at (xi, depth) on the RZ curve, located on vertical plane pl."""
        R, Z = rz(xi, depth)
        c = pl["c"]  # plane offset from the axis
        if R <= abs(c):  # plane cannot be farther from the axis than R
            raise ValueError("Plane offset exceeds local radius.")  # stop with a clear message
        # 3D point: offset along n + the rest of R along m + Z
        return c * pl["n"] + math.sqrt(R * R - c * c) * pl["m"] + Z * EZ


    def path_points(pl, depth, xi_a, xi_b, n=None):  # sweep-path points on a plane at one depth, equal arc spacing
        n = n or Sweep_path_points  # default resolution
        xs = arcmap(depth).equal_arc_xi(xi_a, xi_b, n)  # equal-arc angles
        return [point_on_plane(pl, x, depth) for x in xs], xs  # 3D points + their angles


    def start_frame(pts, xi0, pl, depth):  # profile frame at the start of a sweep path
        """Exact unit tangent T, plane normal n and 3D depth normal N at a path
        start (the profile must be exactly perpendicular to the path)."""
        h = 1.0e-5
        T = point_on_plane(pl, xi0 + h, depth) - point_on_plane(pl, xi0 - h, depth)  # tangent by central difference
        T /= np.linalg.norm(T)  # normalized
        p = pts[0]  # first path point
        phi = math.atan2(p[1], p[0])  # its toroidal angle
        nr = rz_normal(xi0)  # R-Z normal at xi0
        # that normal rotated to the point's toroidal angle
        N3 = np.array([nr[0] * math.cos(phi), nr[0] * math.sin(phi), nr[1]])
        return T, pl["n"], N3  # tangent, plane normal, depth normal


    def orthonormal_in_profile(w, T, u):  # component of w perpendicular to T and u, normalized
        v = w - np.dot(w, T) * T - np.dot(w, u) * u  # remove the T and u parts
        return v / np.linalg.norm(v)  # unit vector


    # swept "square minus quarter circle" that rounds one duct corner to radius r
    def corner_trimmer(pl, depth_face, face_is_front, r, xi_a, xi_b, push=None):
        """Swept body removing the sharp corner between vertical plane `pl`
        (interior on +n side) and the depth face at `depth_face`, leaving an
        R=r fillet. Path = fillet-center curve, extended past the feature ends."""
        d_center = depth_face + r if face_is_front else depth_face - r
        pc = offset_plane(pl, r)  # path plane = side plane moved r into the duct
        xa, xb = extended_xi_range(d_center, xi_a, xi_b, Trimmer_overrun)  # run the path past both duct ends
        pts, xs = path_points(pc, d_center, xa, xb)  # path points on the fillet-center curve
        T, n, N3 = start_frame(pts, xs[0], pc, d_center)  # profile frame at the path start
        u = -n                                     # toward the side plane
        v = orthonormal_in_profile(-N3 if face_is_front else N3, T, u)  # toward the face
        O = pts[0]
        P = lambda U, V: O + U * u + V * v  # profile point at (U toward the side, V toward the face)
        # outer margin and arc push (per-call push overrides the default)
        e, d = Trimmer_outer_margin, (Trimmer_arc_push if push is None else push)
        A, B = P(r + d, 0.0), P(0.0, r + d)  # arc ends, pushed just outside the side plane and the face
        M = P(r / math.sqrt(2.0), r / math.sqrt(2.0))  # arc middle point (exactly on radius r)
        # arc + 4 lines = square minus quarter circle
        segs = [("arc", A, M, B),
                ("line", B, P(0.0, r + e)),
                ("line", P(0.0, r + e), P(r + e, r + e)),
                ("line", P(r + e, r + e), P(r + e, 0.0)),
                ("line", P(r + e, 0.0), A)]
        surf = make_profile_surface(segs)  # planar profile surface
        path, pv = make_spline_path(pts)  # spline path through the points
        return sweep_along(surf, path, pv)  # sweep the profile along the path


    def He_hole(th0, th1, d0, d1, xi_a, xi_b, r, label):  # internal He hole = same construction as a duct
        """Internal He hole built EXACTLY like a PbLi duct: the RZ band d0..d1
        revolved over th0..th1 (radial sides, open/flush at the top and bottom),
        with its 4 poloidal edges rounded R=r by swept corner trimmers
        (sharp corners when He_hole_fillets is False)."""
        return rounded_duct(d0, d1, plane_left(th0), plane_right(th1), th0, th1,
                            xi_a, xi_b, r if He_hole_fillets else 0.0, label)


    # thick prism on the OUTSIDE of a vertical plane (cuts a duct side flat)
    def slab(pl, z_lo, z_hi, m_lo, m_hi, thickness=0.5):
        """Prism covering the region c - thickness <= n.p <= c (outside the plane)."""
        n, m, c = pl["n"], pl["m"], pl["c"]
        # 4 plan corners of the slab
        corners = [c * n + m_lo * m, c * n + m_hi * m, (c - thickness) * n + m_hi * m, (c - thickness) * n + m_lo * m]
        segs = []  # segments
        pts = [np.array([q[0], q[1], z_lo]) for q in corners]  # corners at z_lo
        for i in range(4):  # 4 edges
            segs.append(("line", pts[i], pts[(i + 1) % 4]))  # closed outline
        s = make_profile_surface(segs)  # planar surface
        # Cubit: extrude up in Z
        return run_sweep(f'sweep surface {s} direction 0 0 1 distance {z_hi - z_lo:.12g}', 'side-plane slab')


    # ============================================================================
    # ROUNDED DUCT BODY
    # ============================================================================

    # Z range for the side slabs (covers the whole blanket)
    _Z_SPAN = (-(b + a + total_OB_blanket_depth + 1.0), b + a + total_OB_blanket_depth + 1.0)


    # one PbLi duct / He hole body: revolve, flatten sides, round corners
    def rounded_duct(d0, d1, left_pl, right_pl, th_lo, th_hi, xi_a, xi_b, r, label, round_back=True, push=None):
        """Duct between depths d0..d1, bounded by vertical planes left_pl (interior
        +n) and right_pl (interior +n), filleted R=r on its 4 poloidal edges.
        th_lo..th_hi is the revolve range; it must cover the duct. Planes that are
        exactly the revolve's radial sides (c == 0 at th_lo/th_hi) need no slab."""
        log(f"  duct {label}")
        body = revolve_rz_patch(d0, d1, xi_a, xi_b, th_lo, th_hi - th_lo)  # revolved band d0..d1 over th_lo..th_hi
        slabs = []  # side slabs needed
        Rlo = min_R_over([d0, d1], xi_a, xi_b) - 0.5  # slab extent along the plane (inner)
        Rhi = max_R_over([d0, d1], xi_a, xi_b) + 0.5  # slab extent along the plane (outer)
        for pl, th_edge in ((left_pl, th_lo), (right_pl, th_hi)):  # each side
            if abs(pl["c"]) < 1e-12 and abs(pl["theta"] - th_edge) < 1e-9:  # side already IS the revolve face
                continue  # no slab needed
            slabs.append(slab(pl, _Z_SPAN[0], _Z_SPAN[1], Rlo, Rhi))  # slab that cuts this side flat
        body = subtract(body, slabs, label=f"{label} side planes")  # cut the sides
        if r > 0.0:  # rounded corners requested
            trims = []  # corner trimmers
            for pl in (left_pl, right_pl):  # each side
                trims.append(corner_trimmer(pl, d0, True, r, xi_a, xi_b, push))  # front corner
                if round_back:  # back corners too
                    trims.append(corner_trimmer(pl, d1, False, r, xi_a, xi_b, push))  # back corner
            body = subtract(body, trims, label=f"{label} fillets")  # cut all corners at once
        assert_solid(body, label)  # must be a real solid
        show()  # redraw if graphics are on
        return body  # duct body ID


    # ============================================================================
    # WRAPPED-He CHANNELS
    # ============================================================================

    def wrapped_channel_centers():  # channel centre angles: equal pitch, end walls = half the inter-channel RAFM
        """xi of every channel center. Measured along the curve at the FRONT face of
        the voids (closest to the first wall, where neighbouring channels are closest
        because they splay out along their normals): as many channels as fit at the
        pitch He_channel_height + He_channel_stack_wall (4 mm RAFM = 2 + 2 between
        channels there); the pitch is then stretched slightly so the channels fill
        the band exactly, which makes each end wall exactly HALF the RAFM between
        two channels. Further back the RAFM between channels grows."""
        m = ArcMap(FW_depth + 0.5 * (He_band_thickness - He_void_width), OB_xi_min, OB_xi_max)  # void front face
        L = m.s[-1]  # arc length of the band
        n = int(math.floor(L / He_pitch))  # whole channels that fit at the nominal pitch
        p = L / n  # stretched pitch (>= He_pitch)
        return [m.xi_at((i + 0.5) * p) for i in range(n)], p  # (center angles, pitch)


    def rz_tangent_normal(xi):  # unit R-Z tangent and normal at xi
        """Unit RZ tangent (poloidal) and normal (into the blanket) at xi."""
        p1, p0 = XZcoords(xi + 1.0e-5, 0.0), XZcoords(xi - 1.0e-5, 0.0)
        T = (p1 - p0) / np.linalg.norm(p1 - p0)  # unit tangent
        N = rz_normal(xi)  # unit normal
        return T, N  # both


    def revolve_rz_polygon(pts_rz, th_start, th_sweep, label):  # revolve a closed (R, Z) polygon about the machine axis
        """Revolve a planar polygon given in (R, Z) about the machine axis."""
        P = [np.array([r_, 0.0, z_]) for r_, z_ in pts_rz]
        segs = [("line", P[i], P[(i + 1) % len(P)]) for i in range(len(P))]  # closed outline
        s = make_profile_surface(segs)  # planar surface
        if abs(th_start) > 1e-12:  # start angle not zero
            cubit.cmd(f"rotate surface {s} angle {th_start:.12g} about z")  # Cubit: rotate to the start angle
        return run_sweep(f"sweep surface {s} zaxis angle {th_sweep:.12g}", label)  # Cubit: revolve


    def channel_strip(xi_c):  # revolved strip that picks one channel out of the shell
        """Revolved RZ strip of the channel's poloidal height, bounded by two
        lines PARALLEL to the local normal at xi_c (so the channel keeps the same
        16.12 mm height all the way back), spanning the whole depth."""
        T, N = rz_tangent_normal(xi_c)
        base = XZcoords(xi_c, 0.0)  # plasma-boundary point at the channel center
        h = 0.5 * He_void_height  # half the void height
        d_lo, d_hi = FW_depth - 0.01, Back_depth + 0.05  # depth range (past the FW and the back face)
        # 4 corners: two lines parallel to the local normal
        pts = [base + h * T + d_lo * N, base + h * T + d_hi * N, base - h * T + d_hi * N, base - h * T + d_lo * N]
        # revolve over the sector (+0.5 deg each side)
        return revolve_rz_polygon([(float(p[0]), float(p[1])) for p in pts], -0.5, Sector_angle + 1.0, 'channel strip')


    def intersect_keep_first(a, b, label):  # intersect two volumes, keep the first, delete the second
        """New volume = a AND b; keeps a, deletes b."""
        before = set(cubit.get_entities("volume"))
        cubit.cmd(f"intersect volume {a} {b} keep")  # Cubit: intersection as a new volume
        new = new_volumes_since(before)  # IDs created
        if len(new) != 1:  # must be exactly one
            raise RuntimeError(f"{label}: intersect produced {sorted(new)}")  # stop with a clear message
        cubit.cmd(f"delete volume {b}")  # delete the strip
        assert_solid(new[0], label)  # must be a real solid
        return new[0]  # channel ID


    def build_wrapped_He_shell():  # U band 4..34 mm inside the FW and side walls (all channels come from it)
        """U-shaped band 4..34 mm inside the FW and both side walls (bends
        concentric with the R88.8 FW corners), running out the back face."""
        xa, xb = OB_xi_min, OB_xi_max
        # outer surface of the void band
        outer = rounded_duct(FW_depth + He_wall, Back_depth,
                             plane_left(0.0, He_wall), plane_right(Sector_angle, He_wall),
                             0.0, Sector_angle, xa, xb, He_void_outer_bend_radius,
                             "wrapped-He shell outer", round_back=False)
        # inner surface of the void band
        inner = rounded_duct(FW_depth + He_wall + He_void_width, Back_depth,
                             plane_left(0.0, He_wall + He_void_width),
                             plane_right(Sector_angle, He_wall + He_void_width),
                             0.0, Sector_angle, xa, xb, He_void_inner_bend_radius,
                             "wrapped-He shell inner", round_back=False)
        return subtract(outer, [inner], label="wrapped-He shell")  # band = outer minus inner


    # ============================================================================
    # He HOLE LAYOUTS
    # ============================================================================

    def large_row_layout(hr):  # constant-angle layout of one large-He hole row
        """Constant-angle hole row (same scaling as the ducts): sizes are exact at
        the narrowest radius Rref and grow with R elsewhere, so holes, ribs and
        the side-wall offset are never below their inputs."""
        xa, xb = feature_xi_range(hr["dc"])
        # narrowest radius of the row
        Rref = min_R_over([hr["dc"] - 0.5 * hr["across"], hr["dc"] + 0.5 * hr["across"]], xa, xb)
        # left edge angle after the side offset + end ligament
        thL = math.degrees(math.asin(Side_offset / Rref))  # He band inner face at Rref
        L = Rref * math.radians(Sector_angle - 2.0 * thL)  # usable arc at Rref
        dists, end = fixed_rib_centered(L, hr["along"], hr["rib"], Min_RAFM_ligament, hr["name"])  # centred
        rib = hr["rib"]  # fixed rib
        angles = [thL + math.degrees(s / Rref) for s in dists]  # centers as angles
        half = math.degrees(0.5 * hr["along"] / Rref)  # half the hole width as an angle
        return dict(xi=(xa, xb), Rref=Rref, angles=angles, rib=rib, half_angle=half)  # layout dict


    def small_hole_segments():  # depth stretch for the column-web holes of each PbLi row = the duct length
        """Depth segments (per PbLi row): the row's front and back depth."""
        return [(i, row["d0"], row["d1"]) for i, row in enumerate(PbLi_rows)]  # (row, front depth, back depth)


    def small_hole_plan(d_front, d_back):  # (centre depth, length) of each column hole: fixed rib, centred
        centres, end = fixed_rib_centered(d_back - d_front, small_He_hole_radial_length, small_He_interhole_gap,
                                          Min_RAFM_ligament)
        return [(d_front + c, small_He_hole_radial_length) for c in centres]


    def validate_inputs():  # stop early on impossible OB inputs
        if Num_PbLi_rows < 1:  # need at least 1 row
            raise ValueError("Num_PbLi_rows_OB must be at least 1.")  # stop with a clear message
        # inputs that must be positive
        for name, v in [("PbLi_corner_radius", PbLi_corner_radius),
                        ("web_He_hole_RAFM_ligament", web_He_hole_RAFM_ligament),
                        ("Rear_He_gap_to_ducts", Rear_He_gap_to_ducts),
                        ("PbLi_intercolumn_web_width", PbLi_intercolumn_web_width),
                        ("small_He_hole_corner_radius", small_He_hole_corner_radius)]:
            if v <= 0.0:  # non-positive
                raise ValueError(f"{name} must be > 0.")  # stop naming it
        if OB_extra_back_wall_thickness < 0.0:  # extra back wall cannot be negative
            raise ValueError("OB_extra_back_wall_thickness cannot be negative.")  # stop with a clear message
        # SiC insets must leave a corner radius
        if Build_SiC_inserts and PbLi_corner_radius - SiC_wall_thickness <= 0.0:  # edge-duct SiC inner corner
            raise ValueError("PbLi_duct_fillet_radius must be larger than SiC_thickness.")  # stop with a clear message
        if Build_SiC_inserts and len(SiC_front_gap_by_row) != Num_PbLi_rows:  # one front gap per row
            raise ValueError("SiC_front_gap_by_row needs one value per PbLi row.")  # stop with a clear message
        if small_He_hole_toroidal_width >= PbLi_intercolumn_web_width:  # small hole must fit the web
            raise ValueError("Small He hole does not fit in the column web.")  # stop with a clear message
        # wrapped void must fit the band and corner
        if Build_wrapped_He_channels and (He_wall <= 0.0 or He_void_inner_bend_radius <= 0.0):
            raise ValueError("Wrapped-He void does not fit in the band / FW corner.")  # stop with a clear message
        if FW_corner_outer_radius <= He_band_thickness:  # corner radius must exceed the band
            raise ValueError("FW corner radius must exceed the He band thickness.")  # stop with a clear message
        for row in PbLi_rows:  # each row
            if row["d1"] - row["d0"] <= 2.0 * PbLi_corner_radius:  # row must be deeper than two fillets
                raise ValueError("A PbLi row is too shallow for its corner radius.")  # stop with a clear message


    # ============================================================================
    # MAIN BUILD
    # ============================================================================

    def stage(msg):  # log an outboard stage and (optionally) redraw
        log("OB: " + msg)  # time-stamped message
        show()  # redraw if graphics are on


    validate_inputs()  # stop early on impossible inputs
    layouts = [column_layout(row) for row in PbLi_rows]  # duct layout per row
    hole_layouts = [large_row_layout(hr) for hr in He_hole_rows]  # hole layout per large row

    print("\n==============================================================")  # banner
    print(" FNSF DCLL OUTBOARD")  # title
    print("==============================================================")  # banner
    print(f"  sector {Sector_angle:.3f} deg; xi {OB_xi_min:.3f} .. {OB_xi_max:.3f} deg")  # sector and poloidal range
    print(f"  FW arc length {ArcMap(tSOL, OB_xi_min, OB_xi_max).s[-1]:.3f} m")  # FW arc length
    # total depth and row depths
    print(f"  DERIVED total depth {total_OB_blanket_depth*1000:.2f} mm; PbLi rows "
          + ", ".join(f"{h*1000:.2f}" for h in PbLi_row_depths) + " mm")
    for row, lay in zip(PbLi_rows, layouts):  # each row
        # row summary
        print(f"  row {row['row']+1}: depth {row['d0']-tSOL:.4f}-{row['d1']-tSOL:.4f} m, "
              f"Rref {lay['Rref']:.3f} m, center {lay['center']*1000:.1f} mm, "
              f"edge {lay['edge']*1000:.1f} mm (at Rref)")
    for hr, hl in zip(He_hole_rows, hole_layouts):  # each large-He row
        # row summary
        print(f"  {hr['name']} He row: {len(hl['angles'])} holes, rib {hl['rib']*1000:.2f} mm at "
              f"Rref {hl['Rref']:.3f} m")


    # ---------------------------------------------------------------------------
    # 1. RAFM body + rounded FW corners (trimmed while the body is simple)
    # ---------------------------------------------------------------------------
    stage("RAFM body")
    # revolved RAFM body (FW surface -> back face)
    rafm = revolve_rz_patch(FW_depth, Back_depth, OB_xi_min, OB_xi_max, 0.0, Sector_angle)
    if FW_corner_outer_radius > 0.0:  # round the two FW corners
        # corner trimmers on both sector walls
        trims = [corner_trimmer(plane_left(0.0), FW_depth, True, FW_corner_outer_radius,
                                OB_xi_min, OB_xi_max),
                 corner_trimmer(plane_right(Sector_angle), FW_depth, True, FW_corner_outer_radius,
                                OB_xi_min, OB_xi_max)]
        rafm = subtract(rafm, trims, label="FW corners")  # cut them
    assert_solid(rafm, "RAFM body")  # must be a real solid

    # ---------------------------------------------------------------------------
    # 1b. Wrapped-He channels FIRST (plain body): shell (U band) AND one strip per channel
    # ---------------------------------------------------------------------------
    wrap_ids = []
    if Build_wrapped_He_channels:  # wrapped channels on
        centers_xi, pitch = wrapped_channel_centers()  # channel positions
        # stage message
        stage(f"Wrapped-He channels: {len(centers_xi)} channels (pitch {pitch*1000:.3f} mm at the void front face: "
              f"{(pitch - He_void_height)*1000:.3f} mm RAFM between channels, {0.5*(pitch - He_void_height)*1000:.3f} mm "
              f"end walls)")
        shell = build_wrapped_He_shell()  # U band shell
        for i, xc in enumerate(centers_xi, 1):  # each channel
            # channel = shell AND its strip
            ch = intersect_keep_first(shell, channel_strip(xc), f"wrapped-He channel {i}")
            name_volume(ch, f"OB_HE_WRAP_{i:03d}")  # name it
            wrap_ids.append(ch)  # record it
            if i % 10 == 0 or i == len(centers_xi):  # every 10 channels
                log(f"  wrapped-He channel {i}/{len(centers_xi)} built")  # progress line
                show()  # redraw if graphics are on
        cubit.cmd(f"delete volume {shell}")  # delete the shell
        rafm = subtract_batched(rafm, wrap_ids, "wrapped He")  # cut all channels from the RAFM in batches (kept as He)

    yield "OB phase 1 done"      # body + FW corners + wrapped He

    # ---------------------------------------------------------------------------
    # 2-4. PbLi ducts + SiC inserts, VALIDATED before they are patterned
    #   Each UNIQUE duct (C1, C2 prototype, C9 of every row) is built as
    #   cavity + SiC-outer + SiC-inner layers and split right away into
    #   PbLi gap / SiC shell / PbLi core. Those three bodies get an ACIS
    #   'validate'. If any fails (bad vertex / curve), the whole duct is rebuilt
    #   with a larger corner-trimmer arc push (Duct_trimmer_push_retries), so the
    #   fillet trimmers cut the faces more transversally. Only a clean C2 is then
    #   rotate-copied to C3..C8 (previously one bad C2 was copied 6 times).
    #   A copy of each cavity cuts the RAFM and is then deleted.
    # ---------------------------------------------------------------------------
    _validate_counter = getattr(cubit, "get_error_count", None)
    if not callable(_validate_counter):  # count not available in this Cubit
        # tell the user
        log("  NOTE: cubit.get_error_count() not available; duct validation is "
            "logged by Cubit but cannot trigger an automatic rebuild")


    def volume_is_valid(vid):  # validate a volume; True/False, or None if errors cannot be counted
        """True/False from 'validate volume', or None if it cannot be detected."""
        if not callable(_validate_counter):
            cubit.cmd(f"validate volume {vid}")  # still run validate (output goes to the console)
            return None  # unknown
        n0 = _validate_counter()  # error count before
        cubit.cmd(f"validate volume {vid}")  # Cubit: validate this volume
        return _validate_counter() == n0  # valid if no new errors


    def copy_volume(vid):  # plain copy of a volume (same place)
        before = set(cubit.get_entities("volume"))  # volumes before
        cubit.cmd(f"volume {vid} copy")  # Cubit: copy
        new = new_volumes_since(before)  # IDs created
        if len(new) != 1:  # must be exactly one
            raise RuntimeError(f"Copy of volume {vid} failed.")  # stop with a clear message
        return new[0]  # copy ID


    def layer_corner_and_front(row_i, col, li, ncols):  # (corner radius, front-face inset) of one duct layer
        """Layer 0 = cavity, 1 = SiC outer surface, 2 = SiC inner surface (inboard / reference values)."""
        if li == 0:  # PbLi cavity
            return PbLi_corner_radius, 0.0
        front = SiC_front_gap_by_row[row_i]  # PbLi gap at the front face of this row
        if li == 1:  # SiC outer surface
            return PbLi_corner_radius, front
        edge = col in (0, ncols - 1)  # edge duct?
        r = PbLi_corner_radius - SiC_wall_thickness if edge else PbLi_corner_radius  # SiC inner corner radius
        return r, front + SiC_wall_thickness  # SiC inner surface


    def duct_layers(row, lay, col, push):  # cavity / SiC-outer / SiC-inner bodies of ONE duct
        """Cavity / SiC-outer / SiC-inner bodies of ONE duct (same construction
        as before; only the trimmer arc push can change)."""
        d0, d1 = row["d0"], row["d1"]
        xa, xb = lay["xi"]  # poloidal range
        nC = lay["ncols"]  # columns in this row
        thL_cover = math.degrees(math.asin(Side_offset / lay["Rmax"])) - 0.05  # left edge-duct revolve start angle
        thR_cover = Sector_angle - thL_cover  # right edge-duct revolve end angle
        c0, c1 = lay["cols"][col]  # this column's angle range
        ids = []  # layer volume IDs
        for li, inset in enumerate(_layer_insets):  # cavity, SiC outer, SiC inner
            r, front_inset = layer_corner_and_front(row["row"], col, li, nC)  # corner radius + front-face inset
            dd0, dd1 = d0 + front_inset, d1 - inset  # layer depths (front face may sit further in)
            tag = f"R{row['row']+1}C{col+1} " + ["CAV", "SICO", "SICI"][li]  # label
            if col == 0:  # column 1: outer side parallel to the left sector wall
                # its planes and revolve range
                left, right, tlo, thi = plane_left(0.0, Side_offset + inset), plane_right(c1, inset), thL_cover, c1
            elif col == nC - 1:  # column 9: outer side parallel to the right sector wall
                # its planes and revolve range
                left, right, tlo, thi = plane_left(c0, inset), plane_right(Sector_angle, Side_offset + inset), c0, thR_cover
            else:
                # center column: radial sides
                left, right, tlo, thi = plane_left(c0, inset), plane_right(c1, inset), c0, c1
            ids.append(rounded_duct(dd0, dd1, left, right, tlo, thi, xa, xb, r, tag, push=push))  # build this layer
        return ids  # [cavity, SiC outer, SiC inner]


    def build_checked_duct(row, lay, col):  # build one duct, split it, validate; rebuild with a bigger push if needed
        """Returns {'cutter', 'gap', 'sic', 'core'} (gap/sic/core None without SiC)."""
        tag = f"R{row['row']+1}_C{col+1}"
        # push values to try (per-duct override first)
        tries = list(Duct_trimmer_push_override.get((row['row'] + 1, col + 1), Duct_trimmer_push_retries))
        for k, push in enumerate(tries):  # each attempt
            before_attempt = set(cubit.get_entities("volume"))  # volumes before this attempt
            try:  # a Boolean that fails (e.g. ACIS 'inconsistent face-body relationships') -> next push
                parts, check = duct_attempt(row, lay, col, push, tag)
            except RuntimeError as err:  # this attempt broke
                junk = sorted(set(cubit.get_entities("volume")) - before_attempt)  # whatever it created
                if junk:
                    cubit.cmd("delete volume " + " ".join(map(str, junk)))  # remove it
                if k == len(tries) - 1:  # nothing left to try
                    raise RuntimeError(f"{tag}: every trimmer push failed; last error: {err}")
                log(f"  {tag}: Boolean failed with arc push {push*1e6:.0f} um ({err}) -> "
                    f"rebuilding with {tries[k + 1]*1e6:.0f} um")
                continue  # next push
            if False not in check:  # all passed (or validation unavailable)
                if k > 0 or push != Trimmer_arc_push:  # non-default push used
                    log(f"  {tag}: valid with trimmer arc push {push*1e6:.0f} um")  # report it
                return parts  # done
            if k == len(tries) - 1:  # last attempt failed too
                # warn and keep it
                log(f'  WARNING {tag}: still fails ACIS validate after {len(tries)} tries (kept; check this duct)')
                return parts  # keep it
            # report the rebuild
            log(f'  {tag}: ACIS validate failed -> rebuilding with arc push {tries[k + 1] * 1000000.0:.0f} um')
            # delete this attempt's bodies
            cubit.cmd("delete volume " + " ".join(str(v) for v in parts.values() if v is not None))

    def duct_attempt(row, lay, col, push, tag):  # one build of a duct's layers + split + validate
        if True:
            layers = duct_layers(row, lay, col, push)  # cavity + SiC layers
            if not Build_SiC_inserts:  # no SiC inserts
                ok = volume_is_valid(layers[0])  # validate the cavity
                # the cavity itself is the PbLi body
                parts = {"cutter": layers[0], "gap": None, "sic": None, "core": None}
                check = [ok]  # result
            else:
                cav, so, si = layers  # the three layers
                cutter = copy_volume(cav)  # copy of the cavity: used only to cut the RAFM
                gap = subtract(cav, [so], keep_tool=True, label=f"{tag} PbLi gap")  # PbLi gap = cavity - SiC outer
                sic = subtract(so, [si], keep_tool=True, label=f"{tag} SiC")  # SiC shell = SiC outer - SiC inner
                parts = {"cutter": cutter, "gap": gap, "sic": sic, "core": si}  # the duct's bodies
                check = [volume_is_valid(v) for v in (gap, sic, si)]  # validate gap, SiC and core
            for v in parts.values():  # every body must be a real solid
                if v is not None:
                    assert_solid(v, tag)
            return parts, check


    pbli_ids, sic_ids, cav_cutters = [], [], []  # output lists + cavity copies for the RAFM cut
    for row, lay in zip(PbLi_rows, layouts):  # each PbLi row
        # stage message
        nC = lay["ncols"]  # columns in this row
        stage(f"PbLi row {row['row'] + 1}: {nC} columns (inner {lay['center']*1000:.2f} mm, edge "
              f"{lay['edge']*1000:.2f} mm = {lay['ratio']*100:.1f} %, {Sector_angle/nC:.4f} deg wedges, at R {lay['Rref']*1000:.0f} mm); "
              f"3 unique ducts + rotate copies")
        for col in (0, 1, nC - 1):  # unique columns: first, second (prototype) and last
            d = build_checked_duct(row, lay, col)  # built, split and validated
            entries = [(col, d)]  # (column, bodies) to name and record
            if col == 1:  # column 2 is the prototype
                for c in range(2, nC - 1):  # the other inner columns
                    ang = (c - 1) * lay["pitch"]  # rotation from column 2
                    # rotate-copy every body of the duct
                    entries.append((c, {k: copy_rotate_z(v, ang) if v is not None else None for k, v in d.items()}))
            for c, dd in entries:  # each duct of this group
                tag = f"R{row['row']+1}_C{c+1}"  # label
                cav_cutters.append(dd["cutter"])  # its cavity copy cuts the RAFM
                if Build_SiC_inserts:  # with SiC
                    name_volume(dd["gap"], f"OB_PbLi_gap_{tag}")  # name the PbLi gap
                    name_volume(dd["sic"], f"OB_SiC_{tag}")  # name the SiC shell
                    name_volume(dd["core"], f"OB_PbLi_core_{tag}")  # name the PbLi core
                    pbli_ids += [dd["gap"], dd["core"]]  # record both PbLi bodies
                    sic_ids.append(dd["sic"])  # record the SiC shell
                else:
                    name_volume(dd["cutter"], f"OB_PbLi_{tag}")  # name the PbLi body
                    pbli_ids.append(dd["cutter"])  # record it
            show()  # redraw if graphics are on

    stage("Cutting PbLi cavities from RAFM")  # stage message
    rafm = subtract_batched(rafm, cav_cutters, "PbLi cavities")  # cut all 36 cavities in batches
    if Build_SiC_inserts:  # with SiC the cavity copies are temporary
        cubit.cmd("delete volume " + " ".join(map(str, cav_cutters)))   # cavity copies


    # ---------------------------------------------------------------------------
    # 5. Internal He holes (after the PbLi ducts): built like the ducts
    #    (revolve, optional R3 corner trimmers), one prototype per distinct
    #    hole, rotate-copied about Z
    # ---------------------------------------------------------------------------
    he_ids = list(wrap_ids)
    for hr, hl in zip(He_hole_rows, hole_layouts):  # each large-He row (3 web rows + rear row)
        if not hr["build"]:  # row switched off
            continue  # skip it
        stage(f"{hr['name']} He row: 1 prototype + {len(hl['angles'])-1} rotate copies")  # stage message
        xa, xb = hl["xi"]  # poloidal range
        t0, hw = hl["angles"][0], hl["half_angle"]  # first hole angle and half width
        # prototype hole
        proto = He_hole(t0 - hw, t0 + hw, hr["dc"] - 0.5 * hr["across"], hr["dc"] + 0.5 * hr["across"],
                        xa, xb, hr["r"], f"He {hr['name']} prototype")
        # prototype + rotate copies
        ids = [proto] + [copy_rotate_z(proto, t - hl["angles"][0]) for t in hl["angles"][1:]]
        for i, v in enumerate(ids, 1):  # each hole
            name_volume(v, f"OB_HE_{hr['name']}_{i:03d}")  # name it
        rafm = subtract_batched(rafm, ids, f"{hr['name']} He")  # cut the row in batches (kept as He)
        he_ids += ids  # record them

    if Build_small_He_holes:  # small column-web holes on
        for (ri, d_start, d_end), lay in zip(small_hole_segments(), layouts):  # each PbLi row's depth stretch
            plan = small_hole_plan(d_start, d_end)  # (centre depth, length) of each hole
            if not plan:  # none fit
                print(f"  row {ri+1}: no small holes fit")  # report it
                continue  # next row
            nweb = len(lay["webs"])  # number of webs
            # stage message
            stage(f"Small He holes row {ri+1}: {len(plan)} prototypes x {nweb} webs "
                  f"(lengths {', '.join(sorted(set(f'{hl*1000:.3f}' for _d, hl in plan)))} mm)")
            xa, xb = lay["xi"]  # poloidal range
            th0 = lay["webs"][0]  # angle of web 1
            # same constant-angle scaling as this row's ducts and webs
            hw = math.degrees(0.5 * small_He_hole_toroidal_width / lay["Rref"])
            # One hole per depth (different depths are NOT rigid copies of each
            # other: the curve's normal turns ~51 deg from midplane to the ends),
            # then that hole is rotate-copied to every web (exact, axisymmetric).
            ids = []
            for j, (dc, hl) in enumerate(plan, 1):  # each hole: centre depth and length
                # hole in web 1 at this depth
                p = He_hole(th0 - hw, th0 + hw, dc - 0.5 * hl, dc + 0.5 * hl, xa, xb,
                            small_He_hole_corner_radius, f"He small R{ri+1} depth {j}/{len(plan)}")
                name_volume(p, f"OB_HE_SMALL_R{ri+1}_W1_{j:02d}")  # name it
                ids.append(p)  # record it
                for w in range(1, nweb):  # webs 2..8
                    cp = copy_rotate_z(p, lay["webs"][w] - th0)  # rotate copy
                    name_volume(cp, f"OB_HE_SMALL_R{ri+1}_W{w+1}_{j:02d}")  # name it
                    ids.append(cp)  # record it
                show()  # redraw if graphics are on
            rafm = subtract_batched(rafm, ids, f"small He row {ri+1}")  # cut the row in batches (kept as He)
            he_ids += ids  # record them


    # ---------------------------------------------------------------------------
    # 6. Bookkeeping (blocks, save and export are done by the DRIVER)
    # ---------------------------------------------------------------------------
    name_volume(rafm, "OB_RAFM")
    # phase 2 done: hand the material lists to the driver
    yield {"rafm": [rafm], "he": he_ids, "pbli": pbli_ids, "sic": sic_ids}


# ============================================================================
# DRIVER
# ============================================================================

cubit.cmd("reset")  # start from an empty Cubit model
cubit.cmd("undo off")  # no undo history (saves memory and time)
cubit.cmd("graphics on" if Show_build_in_graphics else "graphics off")  # graphics on/off per Show_build_in_graphics

# The outboard's first step only computes its geometry (nothing is built) and
# reports the Z of its end corners. The inboard spans the FULL outboard
# height = the Z of the OB outer (back) curve at its ends.
ob_gen = build_outboard()
ob_ends = next(ob_gen)  # OB geometry only: returns its end-corner Z values
Blanket_half_height = ob_ends["back_corner_Z"]  # inboard half-height = OB back-corner Z
# report both corner heights
log(f"OB end corners: FW Z = +/- {ob_ends['fw_end_Z']:.4f} m, "
    f"outer (back) curve Z = +/- {ob_ends['back_corner_Z']:.4f} m")
# report the inboard height
log(f"FNSF DCLL IB + OB: inboard height {2*Blanket_half_height*1000:.2f} mm "
    f"(Z = +/- {Blanket_half_height:.4f} m) = full OB outer-curve height")
if not Build_outboard:  # outboard not wanted
    ob_gen.close()  # stop its generator (nothing was built)

parts = []  # (tag, generator) for each blanket built
if Build_inboard:  # inboard on
    parts.append(("IB", build_inboard()))  # add it
if Build_outboard:  # outboard on
    parts.append(("OB", ob_gen))  # add it

# Phase 1 for every part: RAFM body + wrapped-He channels (plain body = fast)
for tag, gen in parts:
    log(f"===== {tag}: RAFM body + wrapped-He channels =====")  # stage banner
    next(gen)  # run to the end of phase 1

# Phase 2 for every part: PbLi ducts + SiC, internal He holes
results = {}
for tag, gen in parts:  # each blanket
    log(f"===== {tag}: PbLi ducts, SiC and internal He holes =====")  # stage banner
    results[tag] = next(gen)  # run to the end; collect its material lists

# ---------------------------------------------------------------------------
# Final checks, blocks, save, export
# ---------------------------------------------------------------------------
log("Final checks")
fam = {k: [] for k in ("rafm", "he", "pbli", "sic")}  # material -> volume IDs (both blankets)
for tag, res in results.items():  # each blanket
    for k in fam:  # each material
        fam[k] += list(res[k])  # append its IDs
all_ids = fam["rafm"] + fam["he"] + fam["pbli"] + fam["sic"]  # every volume
expected = set(all_ids)  # as a set
final = set(cubit.get_entities("volume"))  # volumes that exist in Cubit
if len(all_ids) != len(expected):  # no ID may appear twice
    raise RuntimeError("Duplicate IDs assigned to material families")  # stop with a clear message
if final != expected:  # everything must be assigned, nothing missing
    # stop listing the differences
    raise RuntimeError(f'Volume mismatch: missing={sorted(expected - final)} unassigned={sorted(final - expected)}')
for v in expected:  # every volume
    vol = cubit.volume(v)  # Cubit volume object
    if vol.volume() <= 0 or any(bd.is_sheet_body() for bd in vol.bodies()):  # zero volume or a sheet body
        raise RuntimeError(f"final: volume {v} is not a positive solid")  # stop with a clear message
for tag, res in results.items():  # each blanket
    # its counts
    log(f"  {tag}: RAFM {len(res['rafm'])} | He {len(res['he'])} | PbLi {len(res['pbli'])} | SiC {len(res['sic'])}")
log(f"  TOTAL volumes: {len(expected)}")  # total


def _make_block(ids, name, group):  # block of volumes named 'name' (+ group 'group' for DAGMC)
    if not ids:  # nothing to add
        return  # skip
    bid = cubit.get_next_block_id()  # next free block number
    cubit.cmd(f"block {bid} add volume " + " ".join(map(str, ids)))  # Cubit: add the volumes to the block
    cubit.cmd(f"block {bid} name '{name}'")  # Cubit: name the block
    if Create_OpenMC_material_groups:  # DAGMC groups on
        cubit.cmd(f"group '{group}' add volume " + " ".join(map(str, ids)))  # Cubit: same volumes in group mat:<name>


if Create_blocks:  # blocks on
    _make_block(fam["rafm"], "MAT_RAFM", "mat:RAFM")  # steel
    _make_block(fam["he"], "MAT_HE", "mat:He")  # helium
    _make_block(fam["pbli"], "MAT_PBLI", "mat:PbLi")  # PbLi
    _make_block(fam["sic"], "MAT_SIC", "mat:SiC")  # SiC

cubit.cmd("validate volume all")  # Cubit: ACIS check of every volume (results in the console)
# Cubit has no units. The model is BUILT in metres (the ACIS Boolean tolerances
# the build was tuned with). Cubit's STEP writer labels every number as mm, so
# the STEP files are re-labelled to metres after export (one line in each file;
# the geometry is not touched). The .cub keeps the model's numbers (metres)
# unless Scale_cub_to_mm is True.
def relabel_step_metres(path):  # change the STEP length unit from 'MILLIMETRE' to SI metre (geometry untouched)
    import os, re
    pat = re.compile(r"\(\s*CONVERSION_BASED_UNIT\s*\(\s*'MILLIMETRE'\s*,\s*#\d+\s*\)\s*LENGTH_UNIT\s*\(\s*\)\s*NAMED_UNIT\s*\(\s*(#\d+)\s*\)\s*\)")
    tmp, n = path + ".tmp", 0
    with open(path, "r", errors="ignore", newline="") as fi, open(tmp, "w", newline="") as fo:  # keep line endings
        for line in fi:  # stream: the files are large
            if n == 0 and "MILLIMETRE" in line:
                line, k = pat.subn(lambda m: f"(LENGTH_UNIT()NAMED_UNIT({m.group(1)})SI_UNIT($,.METRE.))", line)
                n += k
            fo.write(line)
    if n != 1:  # not exactly one unit entity found: keep the original file
        os.remove(tmp)
        log(f"  WARNING: STEP unit label not changed in {path} (found {n}); scale it x1000 on import")
        return
    os.replace(tmp, path)
    log(f"  STEP length unit set to metres: {path}")

if Scale_cub_to_mm:
    _probe = sorted(expected)[0]  # check volume for the scale
    _c0 = cubit.get_center_point("volume", _probe)  # its centre before
    cubit.cmd("volume all scale 1000 about 0 0 0")  # Cubit: scale every volume about the origin
    _c1 = cubit.get_center_point("volume", _probe)  # its centre after
    if max(abs(_c1[i] - 1000.0 * _c0[i]) for i in range(3)) > 1.0e-3 * (1.0 + max(map(abs, _c0))):
        raise RuntimeError("Scaling to mm did not scale about the origin; nothing was saved.")
    log("Model scaled x1000 for the .cub (mm)")
log(f"Saving {cubit_file_name}")  # log
cubit.cmd(f"save as '{cubit_file_name}' overwrite")  # Cubit: save the model
if Scale_cub_to_mm:  # STEP files are written from the metre model
    cubit.cmd("volume all scale 0.001 about 0 0 0")  # Cubit: back to metres
if Export_step:  # STEP on
    log(f"Exporting {step_file_name}")  # log
    # Cubit: full model STEP
    cubit.cmd(f"export step '{step_file_name}' volume " + ' '.join(map(str, sorted(expected))) + ' overwrite')
    if STEP_units_metres:
        relabel_step_metres(step_file_name)  # true size in CAD
    # plus one STEP per blanket
    for tag, res in (results.items() if len(results) > 1 else []):  # per-blanket STEP only if both were built
        ids = sorted(res["rafm"] + res["he"] + res["pbli"] + res["sic"])  # this blanket's volumes
        fn = f"{Output_directory}/{Output_basename}_{tag}.stp"  # file name with _IB / _OB
        log(f"Exporting {fn} ({len(ids)} volumes)")  # log
        cubit.cmd(f"export step '{fn}' volume " + " ".join(map(str, ids)) + " overwrite")  # Cubit: this blanket's STEP
        if STEP_units_metres:
            relabel_step_metres(fn)  # true size in CAD
cubit.cmd("graphics on")  # graphics back on to view the result
log("FINISHED")  # log

if progress_log_file:  # log file exists
    try:  # file may be locked
        print("\n================= FULL BUILD LOG =================")  # header
        print(open(progress_log_file).read())  # reprint the whole log (the console can drop lines)
        print("==================================================")  # footer
    except OSError:
        pass  # ignore
