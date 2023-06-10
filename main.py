"""
Colour Checker
KMeans cluster analysis to grab percentages, hex codes and names of colours in a given
image. Simply put any image you'd like to check through the script and see what it says.
author: brendanmoore42@github.com
date: June 2023
"""
import cv2
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from skimage import io
from sklearn.cluster import KMeans

# Main Colour Checker class
class CChecker:
    def __init__(self, image, n_clusters=5, save_image=False):  # number of clusters is n_cluster-1, so 5 is 4
        """Init class, take image, reshape and run KMeans cluster"""
        self.image = image
        self.n_clusters = n_clusters
        self.save_image = save_image

    def analyze_image(self):
        # Run the analysis and output plot
        _, colours = self.run_Kmeans()
        hex_colours = self.fetch_colours(colours)
        self.plot_colours(hex_colours)

    def colour_name(self, hex_code):
        try:  # search color-name.com for match
            r = requests.get(f"https://www.color-name.com/hex/{hex_code[1:]}")
            colour_name = r.text.split("color name is ")[1].split("<")[0]
        except:  # if no colour or site access down, return no name
            print("No name found")
            return "No Name"

        return colour_name

    def fetch_colours(self, colours):
        """Grab colour names, sort and plot"""
        temp_list = []
        for colour in colours:
            temp_list2 = []
            for i in colour[1]:
                temp_list2.append(int(i))
            temp_list.append(temp_list2)

        hex_colours = []
        for j in temp_list:
            hex_code = self.rgb_to_hex(tuple(j))
            hex_colours.append(hex_code)

        return hex_colours

    def plot_colours(self, colours):
        # Create figure, axes
        fig = plt.figure(figsize=[5, 5])
        ax = fig.add_subplot(111)

        # Create rectangles to add to fig
        rect_0 = Rectangle((0, 0), 200, 200, color=colours[0])
        ax.text(66, 100, colours[0])
        temp_name = self.colour_name(colours[0])
        ax.text(66, 80, temp_name)

        rect_1 = Rectangle((0, 0), 200, -200, color=colours[1])
        ax.text(66, -100, colours[1])
        temp_name = self.colour_name(colours[1])
        ax.text(66, -120, temp_name)

        rect_2 = Rectangle((0, 0), -200, 200, color=colours[2])
        ax.text(-132, 100, colours[2])
        temp_name = self.colour_name(colours[2])
        ax.text(-132, 80, temp_name)

        rect_3 = Rectangle((0, 0), -200, -200, color=colours[3])
        ax.text(-132, -100, colours[3])
        temp_name = self.colour_name(colours[3])
        ax.text(-132, -120, temp_name)

        # Add rectangles to fig
        ax.add_patch(rect_0)
        ax.add_patch(rect_1)
        ax.add_patch(rect_2)
        ax.add_patch(rect_3)

        # Clean axes
        plt.xlim([-200, 200])
        plt.ylim([-200, 200])
        plt.axis("off")

        if self.save_image:
            plt.savefig(f"output/{self.image}_analysis.png")
        else:
            plt.show()

    def rgb_to_hex(self, rgb):
        return "#" + "%02x%02x%02x" % rgb

    def run_Kmeans(self):
        image = cv2.imread(self.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        reshape = image.reshape((image.shape[0] * image.shape[1], 3))
        cluster = KMeans(n_clusters=self.n_clusters).fit(reshape)  # n_clusters: how many colours

        visualize, colours = self.visualize_colours(cluster, cluster.cluster_centers_)

        return visualize, colours

    def visualize_colours(self, cluster, centroids):
        # Get clusters, create histogram, normalize values
        labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        (hist, _) = np.histogram(cluster.labels_, bins=labels)
        hist = hist.astype("float")
        hist /= hist.sum()

        # Create frequencies and iterate through each colour and percentage
        rect = np.zeros((50, 300, 3), dtype=np.uint8)
        colours = sorted([(percent, colour) for (percent, colour) in zip(hist, centroids)])

        start = 0
        for (percent, colour) in colours:
            print(colour, "{:0.2f}%".format(percent*100))
            end = start + (percent * 300)
            cv2.rectangle(rect, (int(start), 0), (int(end), 50), colour.astype("uint8").tolist(), -1)
            start = end
        return rect, colours


""" Example: Colloquially referred to as "Blue"and "Red" within the SSBM community, we run an analysis on the two jackets
to find they are in truth Lavender and Orange """

input_image = "input/fox_jacket1.png"
input_image = "input/fox_jacket2.png"

# Set up checker, analyze
CC = CChecker(input_image, n_clusters=5, save_image=True)
CC.analyze_image()
