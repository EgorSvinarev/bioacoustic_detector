import matplotlib.pyplot as plt

def plot_activity(df):
    fig, ax = plt.subplots()
    ax.barh(df["Class"], df["Confidence"])
    ax.set_xlabel("Confidence")
    return fig
