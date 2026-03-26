import numpy as np
import tifffile as tiff
import os
import cv2
from scipy.spatial import Delaunay
import argparse
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path


# ======================
# USER PARAMETERS
# ======================



parser = argparse.ArgumentParser()
parser.add_argument(
    "--image",
    type=str,
    default="example_data/sample_2.tif",
    help="Path to input TIFF file"
)
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, "..", args.image)
filename = os.path.normpath(filename)

start_frame = 8   #0-indexed (8 = 9th image) (selects which frame of tiff file you want to start tracking at)
strain_mode = "ey" #decide which type of strain you want to calculate (ex is x direction strain, ey is y direction strain, gxy is shear strain, and vm is von mises strain)

#triangulation mesh selection
target_triangles = None #integer value for how many triangles you want to track in the mesh 
#(type None for target_triangles if you wish to use mesh_spacing for triangle density instead of target_triangles for number of triangles)

mesh_spacing = 25 #used to determine density of triangles within mesh (lower number is more dense)
#(type None for mesh_spacing if you wish to use target_triangles to designate number of triangles instead of mesh_spacing for triangle density)

show_triangle_edges = True #true = show triangle outlines, false = dont show triangle outlines

delta_x = 20
delta_y = 20
t_x = 15
t_y = 15

low_in = 28
high_in = 250


# ======================
# LOAD TIFF STACK
# ======================

imgs = tiff.imread(filename)

if imgs.ndim == 4:
    imgs = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in imgs])

Nf, mm, nn = imgs.shape

if start_frame < 0 or start_frame >= Nf:
    raise ValueError("start_frame is outside image range.")

def adjust_contrast(img, low, high):
    img = np.clip((img - low) / (high - low), 0, 1)
    return (img * 255).astype(np.uint8)

imgs = np.array([adjust_contrast(frame, low_in, high_in) for frame in imgs])


# ======================
# TRACKING FUNCTION (NCC)
# ======================

def imtrack2_python(img1, img2, x, y, delta_x, delta_y, t_x, t_y):

    target = img1[x:x+delta_x, y:y+delta_y]

    scan_area = img2[
        x - t_x : x + delta_x + t_x,
        y - t_y : y + delta_y + t_y
    ]

    v_t = target.reshape(-1).astype(np.float64)
    mag_t = np.linalg.norm(v_t)

    tempmax = 0
    mx = 0
    my = 0

    for i in range(1 + 2*t_x):
        for j in range(1 + 2*t_y):

            window = scan_area[i:i+delta_x, j:j+delta_y]
            v_s = window.reshape(-1).astype(np.float64)

            mag_s = np.linalg.norm(v_s)
            if mag_s == 0:
                continue

            val = np.dot(v_s, v_t) / (mag_s * mag_t)

            if val > tempmax:
                tempmax = val
                mx = i
                my = j

    rx = -t_x + mx
    ry = -t_y + my

    return rx, ry


# ======================
# ROI SELECTION
# ======================

plt.close('all')
fig, ax = plt.subplots()
ax.imshow(imgs[start_frame], cmap='gray')
ax.set_title(f"Draw ROI on frame {start_frame}\nPress ENTER to finish.")

polygon_vertices = []

def onselect(verts):
    global polygon_vertices
    polygon_vertices = verts

def on_key(event):
    if event.key == 'enter':
        plt.close(fig)

selector = PolygonSelector(ax, onselect)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

if len(polygon_vertices) < 3:
    raise RuntimeError("Polygon not properly drawn.")

roi_path = Path(polygon_vertices)


# ======================
# GENERATE GRID
# ======================

# ======================
# DETERMINE MESH SPACING
# ======================

xmin, ymin = np.min(polygon_vertices, axis=0)
xmax, ymax = np.max(polygon_vertices, axis=0)

# Compute polygon area (shoelace formula)
poly = np.array(polygon_vertices)
x = poly[:,0]
y = poly[:,1]

polygon_area = 0.5 * np.abs(
    np.dot(x, np.roll(y, 1)) -
    np.dot(y, np.roll(x, 1))
)

if target_triangles is not None:

    target_nodes = max(10, int(target_triangles / 2))

    # Estimate spacing
    spacing = int(np.sqrt(polygon_area / target_nodes))

    # Prevent tiny spacing
    spacing = max(8, spacing)

else:

    if mesh_spacing is None:
        raise ValueError("Either target_triangles or mesh_spacing must be set.")

    spacing = mesh_spacing

print(f"Estimated mesh spacing: {spacing}")

points = []

xmin, ymin = np.min(polygon_vertices, axis=0)
xmax, ymax = np.max(polygon_vertices, axis=0)

row_vals = np.arange(int(ymin), int(ymax), spacing)
col_vals = np.arange(int(xmin), int(xmax), spacing)

for r in row_vals:
    for c in col_vals:
        if roi_path.contains_point((c, r)):
            points.append((int(r), int(c)))

Nx = len(points)
if Nx == 0:
    raise RuntimeError("No mesh points generated.")

print(f"Generated {Nx} mesh points.")


# ======================
# INITIALIZE TRACKING
# ======================

matrix_XA = np.full((Nx, Nf), np.nan)
matrix_YA = np.full((Nx, Nf), np.nan)

for i, (row, col) in enumerate(points):
    matrix_XA[i, start_frame] = row
    matrix_YA[i, start_frame] = col


# ======================
# TRIANGULATION
# ======================

pts_array = np.array([[p[1], p[0]] for p in points])
tri = Delaunay(pts_array)


# ======================
# TRACKING LOOP
# ======================

for k in range(start_frame, Nf - 1):
    for i in range(Nx):

        Xi = int(matrix_XA[i, k])
        Yj = int(matrix_YA[i, k])

        Xi = max(t_x, min(Xi, mm - delta_x - t_x - 1))
        Yj = max(t_y, min(Yj, nn - delta_y - t_y - 1))

        rx, ry = imtrack2_python(
            imgs[k], imgs[k+1],
            Xi, Yj,
            delta_x, delta_y,
            t_x, t_y
        )

        matrix_XA[i, k+1] = Xi + rx
        matrix_YA[i, k+1] = Yj + ry


# ======================
# STRAIN (TOTAL RELATIVE TO START FRAME)
# ======================

n_tri = len(tri.simplices)
strain_results = np.full((Nf, n_tri, 3), np.nan)

for k in range(start_frame, Nf):

    for t_id, simplex in enumerate(tri.simplices):

        n1, n2, n3 = simplex

        # Reference positions
        x1_ref = matrix_YA[n1, start_frame]
        y1_ref = matrix_XA[n1, start_frame]
        x2_ref = matrix_YA[n2, start_frame]
        y2_ref = matrix_XA[n2, start_frame]
        x3_ref = matrix_YA[n3, start_frame]
        y3_ref = matrix_XA[n3, start_frame]

        # Current positions
        x1 = matrix_YA[n1, k]
        y1 = matrix_XA[n1, k]
        x2 = matrix_YA[n2, k]
        y2 = matrix_XA[n2, k]
        x3 = matrix_YA[n3, k]
        y3 = matrix_XA[n3, k]

        if np.isnan(x1) or np.isnan(x2) or np.isnan(x3):
            continue

        # Total displacement
        u1 = x1 - x1_ref
        v1 = y1 - y1_ref
        u2 = x2 - x2_ref
        v2 = y2 - y2_ref
        u3 = x3 - x3_ref
        v3 = y3 - y3_ref

        # Reference geometry for B
        x13 = x1_ref - x3_ref
        x23 = x2_ref - x3_ref
        y13 = y1_ref - y3_ref
        y23 = y2_ref - y3_ref

        detJ = x13 * y23 - y13 * x23
        if abs(detJ) < 1e-8:
            continue

        B = (1 / detJ) * np.array([
            [y23, 0, y3_ref - y1_ref, 0, y1_ref - y2_ref, 0],
            [0, x3_ref - x2_ref, 0, x1_ref - x3_ref, 0, x2_ref - x1_ref],
            [x3_ref - x2_ref, y23,
             x1_ref - x3_ref, y3_ref - y1_ref,
             x2_ref - x1_ref, y1_ref - y2_ref]
        ])

        q = np.array([u1, v1, u2, v2, u3, v3])
        strain_results[k, t_id, :] = B @ q


print("Tracking and strain computation complete.")


# ======================
# STRAIN VIEWER
# ======================

plt.close('all')

current_frame = start_frame
fig, ax = plt.subplots()

# ======================
# SELECT STRAIN TYPE
# ======================

if strain_mode == "ex":
    strain_magnitude = strain_results[:, :, 0]
    strain_label = "εx (Horizontal Strain)"

elif strain_mode == "ey":
    strain_magnitude = strain_results[:, :, 1]
    strain_label = "εy (Vertical Strain)"

elif strain_mode == "gxy":
    strain_magnitude = strain_results[:, :, 2]
    strain_label = "γxy (Shear Strain)"

elif strain_mode == "vm":
    ex = strain_results[:, :, 0]
    ey = strain_results[:, :, 1]
    gxy = strain_results[:, :, 2]

    # Von Mises equivalent strain
    strain_magnitude = np.sqrt(
        ex**2 - ex*ey + ey**2 + 0.75*gxy**2
    )

    strain_label = "Von Mises Strain"

else:
    raise ValueError("Invalid strain_mode")

# Percentile-based global scaling (more contrast)
global_min = np.nanpercentile(strain_magnitude, 5)
global_max = np.nanpercentile(strain_magnitude, 95)

if global_max - global_min < 1e-12:
    global_max = global_min + 1e-12

cmap = plt.get_cmap('jet')
norm = plt.Normalize(global_min, global_max)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(strain_label)


def show_frame(k):

    ax.clear()

    img_color = cv2.cvtColor(imgs[k], cv2.COLOR_GRAY2RGB)
    overlay = img_color.copy()

    if k >= start_frame:

        current_pts = np.array([
            [matrix_YA[i, k], matrix_XA[i, k]]
            for i in range(Nx)
        ])

        for t_id, simplex in enumerate(tri.simplices):

            strain_val = strain_magnitude[k, t_id]
            if np.isnan(strain_val):
                continue

            color = cmap(norm(strain_val))
            color_rgb = tuple(int(255 * c) for c in color[:3])

            pts = current_pts[simplex].astype(int)
            cv2.fillPoly(overlay, [pts], color_rgb)
            if show_triangle_edges:
                cv2.polylines(overlay, [pts], True, (0,0,0), 1)

        img_color = cv2.addWeighted(overlay, 0.45, img_color, 0.55, 0)

    ax.imshow(img_color)
    ax.set_title(f"Frame {k} | ← → scroll | q quit")
    fig.canvas.draw_idle()


def on_key(event):
    global current_frame

    if event.key == 'right':
        current_frame = min(current_frame + 1, Nf - 1)
        show_frame(current_frame)

    elif event.key == 'left':
        current_frame = max(current_frame - 1, start_frame)
        show_frame(current_frame)

    elif event.key == 'q':
        plt.close(fig)


fig.canvas.mpl_connect('key_press_event', on_key)

show_frame(current_frame)
plt.show(block=True)