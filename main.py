from flask import Flask, render_template_string, redirect, url_for, render_template
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import os

app = Flask(__name__)
PLOT_FOLDER = "static"
os.makedirs(PLOT_FOLDER, exist_ok=True)


def run_model_and_generate_results():
    # 1. Load and Preprocess Data
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # 2. Simple CNN Model
    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # 3. Compile and Train Model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train_cat,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2,
                        verbose=0)

    # 4. Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)

    # 5. Accuracy/Loss Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    acc_loss_path = os.path.join(PLOT_FOLDER, "accuracy_loss.png")
    plt.savefig(acc_loss_path)

    # 6. Confusion Matrix
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    conf_matrix = confusion_matrix(y_test, y_pred_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = os.path.join(PLOT_FOLDER, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)

    return test_accuracy



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictor')
def predictor():
    try:
        acc = run_model_and_generate_results()
        return render_template('predictor.html', accuracy=f"{acc * 100:.2f}%")
    except Exception as e:
        return f"❌ Error in /predictor: {str(e)}", 500



if __name__ == '__main__':
    app.run(debug=True)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

