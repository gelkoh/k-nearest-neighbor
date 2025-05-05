import argparse, textwrap
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from dataclasses import dataclass
from collections import defaultdict


def run():
    """Main function that stitches everything together and runs the algorithm"""
    args = get_args()

    batches_directory = args.batches_directory
    mode = args.mode
    k = args.k
    training_limit = args.training_limit
    test_limit = args.test_limit
    num_of_plotted_results = args.num_of_plotted_results

    dataset = CIFAR10Dataset(batches_directory)

    knn = KNearestNeighbor(dataset)

    predictions = knn.predict(
        mode,
        k,
        training_limit,
        test_limit
    )

    accuracy = calculate_accuracy(predictions)

    plot_knn_predictions_and_accuracy(predictions, num_of_plotted_results, accuracy)


def get_args():
    """Function retrieves and returns user passed arguments or default arguments 
       if the user did not specify certain arguments"""
    parser = argparse.ArgumentParser(
        description="Implementation of the k-nearest neighbor algorithm on the CIFAR10 dataset",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-bd", "--batches-directory", type=str, 
                        default="./cifar-10-batches-py/",
                        help=textwrap.dedent('''\
                                             Directory where all the batches are located
                                             Default: \"./cifar-10-batches-py/\"
                                             '''))
    parser.add_argument("-m", "--mode", type=str, default="L1",
                        choices=["L1", "L2"],
                        help=textwrap.dedent('''\
                                             Desired distance calculation mode
                                             L1 (Manhattan)-Distance, L2 (Euclidean)-Distance
                                             '''))
    parser.add_argument("-k", type=int, default=5,
                        help="K-neighbors to be accounted in the label prediction")
    parser.add_argument("-trl", "--training-limit", type=int, default=10000,
                        choices=range(1, 50000),
                        metavar="[1-50000]")
    parser.add_argument("-tel", "--test-limit", type=int, default=100,
                        choices=range(1, 10000),
                        metavar="[1-10000]")
    parser.add_argument("-nopr", "--num-of-plotted-results", type=int, 
                        default=10,
                        help="Number of results to be plotted")

    return parser.parse_args()


class CIFAR10Dataset:
    """Class which makes it easy to work with the CIFAR-10 dataset"""
    def __init__(self, directory):
        self.directory = directory
        self.metadata = self._unpickle("batches.meta")
        self.training_data = self._load_training_data()
        self.test_data = self._unpickle("test_batch")


    def _unpickle(self, filename):
        """Method that de-serializes files that were serialized with cPickle and 
           returns the data as a dict"""
        with open(self.directory + filename, "rb") as f:
            dict = pickle.load(f, encoding="bytes")

        return dict


    def _load_training_data(self):
        """Method that stitches the data of the different training data batches 
           together and returns it as one dict"""
        paths = ["data_batch_" + str(i) for i in range(1, 6)]

        labels = []
        data = []
        filenames = []

        for path in paths:
            batch = self._unpickle(path)
            labels += batch[b'labels']
            data.append(batch[b'data'])
            filenames += batch[b'filenames']

        return {
            b'labels': labels,
            b'data': np.vstack(data),
            b'filenames': filenames
        }


class KNearestNeighbor:
    """Class which holds necessary methods to store dataset data and run the 
       knn-algorithm with various options"""
    def __init__(self, dataset):
        self.metadata = dataset.metadata
        self.training_data = dataset.training_data
        self.test_data = dataset.test_data


    def predict(self, mode, k, training_limit, test_limit):
        """Method that applies the knn-algorithm with various options to
           the test data and returns a list of test images with their true and 
           predictes labels"""

        # Normalize values to ensure that they contribute equally to the distance calculations
        self.training_data[b'data'] = self.training_data[b'data'] / 255.0
        self.test_data[b'data'] = self.test_data[b'data'] / 255.0

        training_labels = self.training_data[b'labels'][:training_limit]
        test_labels = self.test_data[b'labels'][:test_limit]
        test_images = self.test_data[b'data'][:test_limit]
        results = []
        distances = []

        for i, test_image in enumerate(test_images):
            # Calculate the distances of the test image to all training images
            distances = self._calc_distances(test_image, mode, training_limit)

            # Get the k-indices of the lowest distances
            lowest_distances_indices = np.argpartition(distances, k)[:k]

            # Get a list of neighbors as tuples containing the label of one of the
            # k-nearest-neighbors to the current test image and its distance to it
            neighbors = [(training_labels[i], distances[i]) for i in lowest_distances_indices]

            # Store the votes per label and the smallest distance to this label
            votes = defaultdict(list)
            for label, distance in neighbors:
                votes[label].append(distance)

            # Select the best label 1. by frequency and 2. by lowest distance
            predicted_label_idx = sorted(
                votes.items(),
                key=lambda item: (-len(item[1]), min(item[1]))
            )[0][0]

            predicted_label = self.metadata[b'label_names'][predicted_label_idx].decode("utf-8")

            true_label = self.metadata[b'label_names'][test_labels[i]].decode("utf-8")

            results.append(ImageWithPredictedLabel(transform_image_data(test_image), 
                                                   str(predicted_label),
                                                   str(true_label)))

        return results


    def _calc_distances(self, test_image, mode, training_limit):
        """Calculate the distances between the given test image and all training 
        images (up to the training limit) based on the specified distance mode."""

        # L1 = Manhattan distance
        if mode == "L1":
            return np.sum(np.abs(self.training_data[b'data'][:training_limit] - test_image), axis=1)

        # L2 = Euclidean distance
        elif mode == "L2":
            return np.linalg.norm(self.training_data[b'data'][:training_limit] - test_image, axis=1)


@dataclass
class ImageWithPredictedLabel:
    """Little data class which makes passing images easier"""
    data: np.ndarray
    predicted_label: str
    true_label: str


def transform_image_data(image_data):
    """Function that transforms a 1D array into a 3D color image array and
       returns it"""
    width = 32
    height = 32
    channel_size = width * height

    r = image_data[:channel_size].reshape(width, height)
    g = image_data[channel_size:channel_size * 2].reshape(width, height)
    b = image_data[channel_size * 2:].reshape(width, height)

    return np.dstack((r, g, b))


def calculate_accuracy(predictions):
    """Function that calculates the total accuracy based on the true labels
       and predicted labels of the used test images"""
    correct = sum(p.predicted_label == p.true_label for p in predictions)
    total = len(predictions)
    return correct / total * 100


def plot_knn_predictions_and_accuracy(predictions, num, accuracy):
    """Function that plots the test images with the predicted labels as well
       as the accuracy of all predicted labels"""
    row_count = num // 5
    if num % 5 != 0: 
        row_count += 1

    row_count = int(row_count)

    fig, axs = plt.subplots(row_count, 5, figsize=(12, 5))
    fig.subplots_adjust(hspace=1)
    fig.canvas.manager.set_window_title("K-Nearest Neighbor")

    axs = axs.reshape(-1)

    for i, ax in enumerate(axs):
        if i < len(predictions) and i < num:
            ax.imshow(predictions[i].data)
            ax.set_title("Label: " + predictions[i].predicted_label)
            ax.axis("off")
        else:
            ax.axis("off")

    fig.text(0.5, 0.02, f"Accuracy: {accuracy:.2f}%", ha="center", fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


if __name__ == "__main__":
    """Entry point"""
    run()
