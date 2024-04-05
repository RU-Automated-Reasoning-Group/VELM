
import numpy as np
arr = [(129, 22), (16, 1), (71, 1), (3319, 4), (22, 1), (253, 1), (7, 1), (1443, 1)]

rst = []
for x, y in arr:
    rst.append(x / y)

print(np.mean(rst))

import matplotlib.pyplot as plt
import matplotlib
ax = plt.gca()
ax.add_patch(matplotlib.patches.Rectangle((0, 0), 4, 6, facecolor="green",alpha=0.15, ec="green", lw=2))
ax.add_patch(matplotlib.patches.Rectangle((-0.5, -0.5), 4, 6, facecolor="green",alpha=0.15, ec="green", lw=2))
ax.add_patch(matplotlib.patches.Rectangle((-1, -1), 4, 6, facecolor="green",alpha=0.15, ec="green", lw=2))
ax.add_patch(matplotlib.patches.Rectangle((-1.5, -1.5), 4, 6, facecolor="green",alpha=0.15, ec="green", lw=2))
ax.add_patch(matplotlib.patches.Rectangle((-2, -2), 4, 6, facecolor="green",alpha=0.15, ec="green", lw=2))
ax.set_xlim([-15, 15])
ax.set_ylim([-15, 15])
plt.savefig("test.png")
# plt.plot([0.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 1.0], color="green")