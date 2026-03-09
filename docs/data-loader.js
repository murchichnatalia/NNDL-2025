data-loader.js
```javascript
// data-loader.js – CSV parsing, tensor creation, noise, drawing, MSE
// no external libraries, FileReader + TextDecoder, careful disposal

/**
 * Parse CSV file (no header) using FileReader.
 * Each row: label (int) + 784 pixel values (0-255)
 * Returns { xs, ys } where xs is [N,28,28,1] normalized, ys one-hot depth 10.
 */
async function loadTrainFromFiles(file) {
    return loadCsvToTensors(file, true);  // true = one-hot labels
}

async function loadTestFromFiles(file) {
    return loadCsvToTensors(file, true);
}

/**
 * Core CSV loader: reads file, splits lines, converts to tensors.
 */
async function loadCsvToTensors(file, oneHot = true) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const text = e.target.result;
                const lines = text.split(/\r?\n/).filter(line => line.trim() !== '');
                const numRows = lines.length;
                if (numRows === 0) throw new Error('empty file');

                const labels = [];
                const pixels = [];

                for (let line of lines) {
                    const values = line.split(',').map(Number);
                    if (values.length !== 785) continue; // safety: label + 784
                    const label = values[0];
                    const image = values.slice(1, 785); // length 784
                    if (label < 0 || label > 9) throw new Error('invalid label');
                    labels.push(label);
                    pixels.push(...image); // flatten all images
                }

                // shape: [numRows, 784] -> normalize /255
                const xsData = new Float32Array(pixels.map(p => p / 255.0));
                const xs = tf.tensor4d(xsData, [numRows, 28, 28, 1]); // [N,28,28,1]

                // one-hot encode labels
                const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 10);

                // small memory cleanup: xs and ys will be returned, caller disposes
                resolve({ xs, ys });
            } catch (err) {
                reject(err);
            }
        };
        reader.onerror = () => reject(new Error('file read failed'));
        reader.readAsText(file); // acceptable for MNIST CSV (~110MB for train)
    });
}

/**
 * Split training tensors into train/validation.
 * returns { trainXs, trainYs, valXs, valYs } (all tensors)
 */
function splitTrainVal(xs, ys, valRatio = 0.1) {
    const total = xs.shape[0];
    const valSize = Math.floor(total * valRatio);
    const trainSize = total - valSize;

    return tf.tidy(() => {
        // random shuffle indices (deterministic seed optional)
        const indices = tf.util.createShuffledIndices(total);
        const trainIndices = indices.slice(0, trainSize);
        const valIndices = indices.slice(trainSize);

        const trainXs = tf.gather(xs, trainIndices);
        const trainYs = tf.gather(ys, trainIndices);
        const valXs = tf.gather(xs, valIndices);
        const valYs = tf.gather(ys, valIndices);

        return { trainXs, trainYs, valXs, valYs };
    });
}

/**
 * Get k random test samples (tensors) for preview.
 * returns { xs: [k,28,28,1], ys: [k,10] } (fresh tensors)
 */
function getRandomTestBatch(xs, ys, k = 5) {
    return tf.tidy(() => {
        const total = xs.shape[0];
        const indices = [];
        for (let i = 0; i < k; i++) {
            indices.push(Math.floor(Math.random() * total));
        }
        const batchXs = tf.gather(xs, indices);
        const batchYs = tf.gather(ys, indices);
        return { xs: batchXs, ys: batchYs };
    });
}

/**
 * Add random Gaussian noise (clipped 0-1) to a tensor.
 * noiseFactor = standard deviation multiplier (e.g., 0.3)
 */
function addRandomNoise(tensor, noiseFactor = 0.3) {
    return tf.tidy(() => {
        const noise = tf.randomNormal(tensor.shape, 0, noiseFactor);
        const noisy = tensor.add(noise);
        return noisy.clipByValue(0, 1); // keep pixel range
    });
}

/**
 * Draw a 28x28 grayscale tensor onto a canvas (scaled).
 */
function draw28x28ToCanvas(tensor, canvas, scale = 4) {
    return tf.tidy(() => {
        const data = tensor.squeeze(); // [28,28] or [1,28,28] -> [28,28]
        const width = 28 * scale;
        const height = 28 * scale;
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(width, height);
        const pixels = data.dataSync(); // float32 0-1

        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                const val = Math.floor(pixels[y * 28 + x] * 255);
                for (let dy = 0; dy < scale; dy++) {
                    for (let dx = 0; dx < scale; dx++) {
                        const px = x * scale + dx;
                        const py = y * scale + dy;
                        const idx = (py * width + px) * 4;
                        imageData.data[idx] = val;
                        imageData.data[idx + 1] = val;
                        imageData.data[idx + 2] = val;
                        imageData.data[idx + 3] = 255;
                    }
                }
            }
        }
        ctx.putImageData(imageData, 0, 0);
    });
}

/**
 * Calculate Mean Squared Error between two tensors (original and reconstructed).
 * Returns scalar (number).
 */
async function calculateMSE(original, reconstructed) {
    return tf.tidy(() => {
        const diff = original.sub(reconstructed);
        const squared = diff.square();
        const mean = squared.mean();
        return mean.dataSync()[0];
    });
}

// expose globally for app.js
window.dataLoader = {
    loadTrainFromFiles,
    loadTestFromFiles,
    splitTrainVal,
    getRandomTestBatch,
    addRandomNoise,
    draw28x28ToCanvas,
    calculateMSE
};