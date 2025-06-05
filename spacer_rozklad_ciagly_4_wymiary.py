import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

steps = 1000
walks = 20
interval = 50

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
colors2d = [plt.cm.plasma(i / walks) for i in range(walks)]
for i in range(walks):
    x = np.cumsum(np.random.uniform(-0.3, 0.3, size=steps))
    y = np.cumsum(np.random.uniform(-0.3, 0.3, size=steps))
    plt.plot(x, y, label=f"Trajektoria {i+1}", linewidth=0.8, alpha=0.6, color=colors2d[i])
plt.title("Spacer losowy w 2D")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(122, projection='3d')
colors3d = [plt.cm.viridis(i / walks) for i in range(walks)]
trajectories3d = []
for i in range(walks):
    x = np.cumsum(np.random.uniform(-0.3, 0.3, size=steps))
    y = np.cumsum(np.random.uniform(-0.3, 0.3, size=steps))
    z = np.cumsum(np.random.uniform(-0.3, 0.3, size=steps))
    trajectories3d.append((x, y, z))
    ax.plot(x, y, z, label=f"Trajektoria {i+1}", linewidth=0.8, alpha=0.6, color=colors3d[i])
all_x = np.concatenate([x for x, y, z in trajectories3d])
all_y = np.concatenate([y for x, y, z in trajectories3d])
all_z = np.concatenate([z for x, y, z in trajectories3d])
margin = 2
ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
ax.set_zlim(all_z.min() - margin, all_z.max() + margin)
ax.set_title("Spacer losowy w 3D")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

fig2d, ax2d = plt.subplots(figsize=(6, 6))
colors2d = [plt.cm.plasma(i / walks) for i in range(walks)]
trajectories2d = []
for _ in range(walks):
    dx = np.random.uniform(-0.3, 0.3, size=steps)
    dy = np.random.uniform(-0.3, 0.3, size=steps)
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    trajectories2d.append((x, y))
all_x2d = np.concatenate([x for x, y in trajectories2d])
all_y2d = np.concatenate([y for x, y in trajectories2d])
margin = 2
ax2d.set_xlim(all_x2d.min() - margin, all_x2d.max() + margin)
ax2d.set_ylim(all_y2d.min() - margin, all_y2d.max() + margin)
ax2d.set_title("Animacja spaceru losowego 2D")
ax2d.set_xlabel("X")
ax2d.set_ylabel("Y")

lines2d = [ax2d.plot([], [], lw=0.8, alpha=0.6, color=colors2d[i])[0] for i in range(walks)]
starts2d = [ax2d.plot([], [], 'o', markersize=2, color=colors2d[i])[0] for i in range(walks)]
ends2d = [ax2d.plot([], [], 'o', markersize=2, color=colors2d[i])[0] for i in range(walks)]

def update_2d(frame):
    for i, (line, start, end) in enumerate(zip(lines2d, starts2d, ends2d)):
        x, y = trajectories2d[i]
        line.set_data(x[:frame], y[:frame])
        start.set_data([x[0]], [y[0]])
        end.set_data([x[min(frame - 1, steps - 1)]], [y[min(frame - 1, steps - 1)]])
    return lines2d + starts2d + ends2d

ani2d = FuncAnimation(fig2d, update_2d, frames=steps, interval=interval, blit=True)
ani2d.save("spacer_2d.mp4", writer=FFMpegWriter(fps=1000 // interval))

fig3d = plt.figure(figsize=(8, 6))
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax3d.set_ylim(all_y.min() - margin, all_y.max() + margin)
ax3d.set_zlim(all_z.min() - margin, all_z.max() + margin)
ax3d.set_title("Animacja spaceru losowego 3D")
ax3d.set_xlabel("X")
ax3d.set_ylabel("Y")
ax3d.set_zlabel("Z")

lines3d = [ax3d.plot([], [], [], lw=0.8, alpha=0.6, color=colors3d[i])[0] for i in range(walks)]
starts3d = [ax3d.plot([0], [0], [0], 'o', markersize=2, color=colors3d[i])[0] for i in range(walks)]
ends3d = [ax3d.plot([], [], [], 'o', markersize=2, color=colors3d[i])[0] for i in range(walks)]

def update_3d(frame):
    for i, (line, start, end) in enumerate(zip(lines3d, starts3d, ends3d)):
        x, y, z = trajectories3d[i]
        line.set_data(x[:frame], y[:frame])
        line.set_3d_properties(z[:frame])
        start.set_data_3d([x[0]], [y[0]], [z[0]])
        end.set_data_3d([x[min(frame - 1, steps - 1)]],
                        [y[min(frame - 1, steps - 1)]],
                        [z[min(frame - 1, steps - 1)]])
    return lines3d + starts3d + ends3d

ani3d = FuncAnimation(fig3d, update_3d, frames=steps, interval=interval, blit=True)
ani3d.save("spacer_3d.mp4", writer=FFMpegWriter(fps=1000 // interval))

plt.close('all')
print("trajektorie i animacje zapisane.")

def random_unit_vector(n):
    vec = np.random.normal(size=n)
    return vec / np.linalg.norm(vec)

def random_walk_nd(dim, steps):
    traj = np.zeros((steps, dim))
    for i in range(1, steps):
        traj[i] = traj[i-1] + random_unit_vector(dim)
    return traj

traj4d = random_walk_nd(4, steps)
x4d, y4d, z4d, w4d = traj4d[:, 0], traj4d[:, 1], traj4d[:, 2], traj4d[:, 3]

fig_anim_4d = plt.figure(figsize=(8, 6))
ax_anim_4d = fig_anim_4d.add_subplot(111, projection='3d')
ax_anim_4d.set_xlim(np.min(x4d)-1, np.max(x4d)+1)
ax_anim_4d.set_ylim(np.min(y4d)-1, np.max(y4d)+1)
ax_anim_4d.set_zlim(np.min(z4d)-1, np.max(z4d)+1)
ax_anim_4d.set_title("Animacja spaceru losowego 4D (rzut na 3D z trajektorią)")
ax_anim_4d.set_xlabel("X")
ax_anim_4d.set_ylabel("Y")
ax_anim_4d.set_zlabel("Z")

line_4d, = ax_anim_4d.plot([], [], [], lw=2, color='blue', alpha=0.5)
point_4d = ax_anim_4d.scatter([], [], [], c=[], cmap='plasma', s=20, vmin=np.min(w4d), vmax=np.max(w4d))

cbar = plt.colorbar(point_4d, ax=ax_anim_4d, label='Wymiar 4 (kolor)')
ax_anim_4d.text2D(0.5, 0.95, "Kolor reprezentuje wartość czwartego wymiaru (W)", transform=ax_anim_4d.transAxes,
                  horizontalalignment='center', fontsize=12, color='black')

def update_4d(frame):
    line_4d.set_data(x4d[:frame], y4d[:frame])
    line_4d.set_3d_properties(z4d[:frame])
    line_4d.set_color(plt.cm.plasma((w4d[frame] - np.min(w4d)) / (np.max(w4d) - np.min(w4d))))

    point_4d._offsets3d = ([x4d[frame]], [y4d[frame]], [z4d[frame]])
    point_4d.set_array(np.array([w4d[frame]]))

    return line_4d, point_4d

ani_4d = FuncAnimation(fig_anim_4d, update_4d, frames=steps, interval=interval, blit=True)

ani_4d.save("spacer_4d.mp4", writer=FFMpegWriter(fps=1000 // interval))

plt.close(fig_anim_4d)
print("Animacja spaceru 4D zapisane.")

trajectories4d = [random_walk_nd(4, steps) for _ in range(walks)]

fig_static_4d = plt.figure(figsize=(10, 8))
ax_static_4d = fig_static_4d.add_subplot(111, projection='3d')
ax_static_4d.set_title("20 trajektorii spaceru losowego 4D (rzut 3D + kolor = 4. wymiar)")
ax_static_4d.set_xlabel("X")
ax_static_4d.set_ylabel("Y")
ax_static_4d.set_zlabel("Z")

all_x, all_y, all_z, all_w = [], [], [], []
for traj in trajectories4d:
    x, y, z, w = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
    all_x.append(x)
    all_y.append(y)
    all_z.append(z)
    all_w.append(w)
    ax_static_4d.plot(x, y, z, alpha=0.6, lw=0.8, color=plt.cm.plasma((w[-1] - np.min(w)) / (np.max(w) - np.min(w))))

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(np.min(w4d), np.max(w4d)))
cbar = plt.colorbar(sm, ax=ax_static_4d, pad=0.1)
cbar.set_label('4. wymiar (W)', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()
