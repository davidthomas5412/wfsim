import pickle
import matplotlib.pyplot as plt

with open("chips.pkl", 'rb') as f:
    chips = pickle.load(f)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
for k, v in chips.items():
    if 'S11' in k:
        ax.text(v['x'], v['y'], k[:3], fontsize=20)
ax.set_xlim(0.35, -0.35)  # reverse, b/c camera coord sys is left handed
ax.set_ylim(-0.35, 0.35)
ax.set_title("Looking through L3 to FP")
ax.set_xlabel(" <--- Camera Coordinate System x")
ax.set_ylabel("Camera Coordinate System y --->")
plt.show()
