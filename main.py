import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras._tf_keras.keras import utils, Sequential
from keras._tf_keras.keras.optimizers import Adam, SGD
from keras._tf_keras.keras.datasets import cifar10
from keras._tf_keras.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D
from keras._tf_keras.keras.layers import Dropout, BatchNormalization
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator


def prepare_datasets():
    """Load dataset and partition train, test and validation sets"""
    (X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

    # partition a validation set from the provided training dataset,
    # we'll split it to a 80/20 ratio for train/validation
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42,
    )

    # visualize our label distribution in train, validation, and test sets
    plot_distributions(y_train, y_validate, y_test)

    # encode data categories
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    y_validate = utils.to_categorical(y_validate)

    # normalize image data, first convert from integers to floats
    train_norm = X_train.astype("float32")
    test_norm = X_test.astype("float32")
    validate_norm = X_validate.astype("float32")

    # next, normalize to range 0-1
    X_train = train_norm / 255.0
    X_test = test_norm / 255.0
    X_validate = validate_norm / 255.0

    return X_train, y_train, X_validate, y_validate, X_test, y_test


def construct_model_vgg(learning_rate=0.001, dropout_rates=[0, 0, 0, 0, 0],
    batch_normalization=False, image_generation=False):
    model = Sequential()
    model.add(Input(shape=(32, 32, 3)))

    # block 1
    # note, in this scheme RelU is applied *before* BN which is technically
    # incorrect, however, in testing activation then normalisation
    # actually lead to better restults?
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    if batch_normalization: model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    if batch_normalization: model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    if dropout_rates[0] > 0: model.add(Dropout(dropout_rates[0]))

    # block 2
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    if batch_normalization: model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    if batch_normalization: model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    if dropout_rates[1] > 0: model.add(Dropout(dropout_rates[1]))

    # block 3
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    if batch_normalization: model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    if batch_normalization: model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    if dropout_rates[2] > 0: model.add(Dropout(dropout_rates[2]))

    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    if batch_normalization: model.add(BatchNormalization())
    if dropout_rates[3] > 0: model.add(Dropout(dropout_rates[3]))

    model.add(Dense(512, activation="relu"))
    if batch_normalization: model.add(BatchNormalization())
    if dropout_rates[4] > 0: model.add(Dropout(dropout_rates[4]))

    # output layer (for CIFAR-10 classification) : 10 classes
    model.add(Dense(10, activation="softmax"))

    # opt = SGD(learning_rate=0.001, momentum=0.8)
    opt = Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    return {
        "model": model,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rates,
        "batch_normalization": batch_normalization,
        "image_generation": image_generation,
    }


def show_learning_diagnostics(history, test_accuracy, test_loss, time_str):
    print(f"Training Time = {time_str}")
    print(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")

    plt.subplot(211)
    plt.plot(history.history["loss"], color="blue", label="training")
    plt.plot(history.history["val_loss"], color="orange", label="validation")
    plt.ylabel(f"Loss (Test:{test_loss:.4f})")
    plt.xticks([])
    plt.legend()

    plt.subplot(212)
    plt.plot(history.history["accuracy"], color="blue", label="training")
    plt.plot(history.history["val_accuracy"], color="orange", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel(f"Accuracy (Test:{test_accuracy:.4f})")

    plt.show()


def plot_distributions(y_train, y_val, y_test):
    y_train = y_train.flatten()
    y_val = y_val.flatten()
    y_test = y_test.flatten()

    train_counts = Counter(y_train)
    val_counts = Counter(y_val)
    test_counts = Counter(y_test)

    # order the labels (0–9 for CIFAR-10)
    labels = sorted(set(y_train))

    data = {
        "Label": labels * 3,
        "Count": [train_counts[l] for l in labels]
        + [val_counts[l] for l in labels]
        + [test_counts[l] for l in labels],
        "Dataset": ["Train"] * len(labels)
        + ["Validation"] * len(labels)
        + ["Test"] * len(labels),
    }

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Label", y="Count", hue="Dataset", data=df)
    # plt.title('Label Distribution in Train, Validation, and Test Sets (CIFAR-10)')
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.legend(title="Dataset")
    plt.show()


def train(model, X_train, y_train, X_validate, y_validate, X_test,
          y_test, use_image_gen, epocs=100):
    
    # we'll use batches of 64 - this could be changed to 32 too
    batch_size = 64

    # capture time for training time statistics
    start_time = time.time()

    if use_image_gen:
        generator = ImageDataGenerator(
            width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
        )
        steps = int(X_train.shape[0] / batch_size)
        history = model.fit(
            generator.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=steps,
            epochs=epocs,
            validation_data=(X_validate, y_validate),
            verbose=0,
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            epochs=epocs,
            batch_size=batch_size,
            validation_data=(X_validate, y_validate),
            verbose=0,
        )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    end_time = time.time()
    total_training_time = end_time - start_time

    return {
        "acc": acc,
        "loss": loss,
        "time": str(datetime.timedelta(seconds=total_training_time)),
        "history": history,
    }


def exhaustive_test_run():
    # load and pre-process our image data
    X_train, y_train, X_validate, y_validate, X_test, y_test = prepare_datasets()

    testcases = [
        {"learning_rate": 0.001, "dropout_rates": [0, 0, 0, 0, 0]},
        {"learning_rate": 0.001, "dropout_rates": [0, 0, 0, 0.5, 0.5]},
        {"learning_rate": 0.001, "dropout_rates": [0.2, 0.2, 0.2, 0.5, 0.5]},
        {"learning_rate": 0.001, "dropout_rates": [0.2, 0.35, 0.5, 0.5, 0.5]},
        {"learning_rate": 0.001, "dropout_rates": [0.2, 0.35, 0.5, 0.6, 0.7]},
        {"learning_rate": 0.0005, "dropout_rates": [0, 0, 0, 0, 0]},
        {"learning_rate": 0.0005, "dropout_rates": [0, 0, 0, 0.5, 0.5]},
        {"learning_rate": 0.0005, "dropout_rates": [0.2, 0.2, 0.2, 0.5, 0.5]},
        {"learning_rate": 0.0005, "dropout_rates": [0.2, 0.35, 0.5, 0.5, 0.5]},
        {"learning_rate": 0.0005, "dropout_rates": [0.2, 0.35, 0.5, 0.6, 0.7]},
        {"learning_rate": 0.0001, "dropout_rates": [0, 0, 0, 0, 0]},
        {"learning_rate": 0.0001, "dropout_rates": [0, 0, 0, 0.5, 0.5]},
        {"learning_rate": 0.0001, "dropout_rates": [0.2, 0.2, 0.2, 0.5, 0.5]},
        {"learning_rate": 0.0001, "dropout_rates": [0.2, 0.35, 0.5, 0.5, 0.5]},
        {"learning_rate": 0.0001, "dropout_rates": [0.2, 0.35, 0.5, 0.6, 0.7]},
    ]

    # build test models
    models = []
    # without batch normalization or image generation
    for case in testcases:
        models.append(
            construct_model_vgg(
                learning_rate=case["learning_rate"], dropout_rates=case["dropout_rates"],
                batch_normalization=False, image_generation=False,
            )
        )

    # with batch normalization but not image generation
    for case in testcases:
        models.append(
            construct_model_vgg( 
                learning_rate=case["learning_rate"], dropout_rates=case["dropout_rates"],
                batch_normalization=True, image_generation=False,
            )
        )

    # without batch normalization but with image generation
    for case in testcases:
        models.append(
            construct_model_vgg(
                learning_rate=case["learning_rate"], dropout_rates=case["dropout_rates"],
                batch_normalization=False, image_generation=True,
            )
        )

    # with batch normalization and with image generation
    for case in testcases:
        models.append(
            construct_model_vgg(
                learning_rate=case["learning_rate"], dropout_rates=case["dropout_rates"],
                batch_normalization=True, image_generation=True,
            )
        )

    index = 1
    count = len(models)
    while len(models) > 0:
        print(f"Processing model {index} of {count}")
        index = index + 1
        model = models.pop(0)
        result = train(
            model["model"],
            X_train,
            y_train,
            X_validate,
            y_validate,
            X_test,
            y_test,
            use_image_gen=model["image_generation"],
            epocs=100,
        )

        print(f"Parameters: {model}")
        show_learning_diagnostics(
            result["history"], result["acc"], result["loss"], result["time"]
        )


def top5_test_run():
    # load and pre-process our image data
    X_train, y_train, X_validate, y_validate, X_test, y_test = prepare_datasets()

    testcases = [
        {"learning_rate": 0.001, "dropout_rates": [0, 0, 0, 0.5, 0.5]},  # FC-Only
        {"learning_rate": 0.001, "dropout_rates": [0.2, 0.2, 0.2, 0.5, 0.5]},  # Linear
        {"learning_rate": 0.0005, "dropout_rates": [0, 0, 0, 0.5, 0.5]},  # FC-Only
        { "learning_rate": 0.0005, "dropout_rates": [0.2, 0.35, 0.5, 0.5, 0.5]},  # Inc-Low
        { "learning_rate": 0.0005, "dropout_rates": [0.2, 0.35, 0.5, 0.6, 0.7]},  # Inc-High
    ]

    # build test models
    models = []
    # with batch normalization and with image generation
    for case in testcases:
        models.append(
            construct_model_vgg(
                learning_rate=case["learning_rate"], dropout_rates=case["dropout_rates"],
                batch_normalization=True, image_generation=True,
            )
        )

    for i, model in enumerate(models):
        print(f"Processing model {i + 1} of {len(models)}")
        result = train(
            model["model"],
            X_train,
            y_train,
            X_validate,
            y_validate,
            X_test,
            y_test,
            use_image_gen=model["image_generation"],
            epocs=300,
        )

        print(f"Parameters: {model}")
        show_learning_diagnostics(
            result["history"], result["acc"], result["loss"], result["time"]
        )


def main():
    # run an exhaustive test run to test the best conbinations of LR,
    # Batch Normalisation, ImageGen and Dropout
    # exhastive_test_run()

    # run the top 5 as determined by the exhaustive run
    top5_test_run()


if __name__ == "__main__":
    main()
