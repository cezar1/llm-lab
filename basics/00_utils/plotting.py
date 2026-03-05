
import matplotlib.pyplot as plt
def heatmap(matrix, title="Heatmap"):
    plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.show()
