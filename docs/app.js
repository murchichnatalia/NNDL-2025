app.js
```javascript
// app.js – full UI wiring, autoencoder models, pooling comparison, save/load
// global state
let trainXs, trainYs, testXs, testYs;          // raw loaded tensors
let classificationModel, modelMax, modelAvg;    // three possible models
let currentMode = 'classification';              // from radio
let noiseLevel = 0.3;                            // from slider

// UI elements
const trainFile = document.getElementById('trainFileInput');
const testFile = document.getElementById('testFileInput');
const loadDataBtn = document.getElementById('loadDataBtn');
const dataStatus = document.getElementById('dataStatus');
const trainBtn = document.getElementById('trainBtn');
const evaluateBtn = document.getElementById('evaluateBtn');
const testFiveBtn = document.getElementById('testFiveBtn');
const comparePoolingBtn = document.getElementById('comparePoolingBtn');
const saveBtn = document.getElementById('saveBtn');
const loadModelBtn = document.getElementById('loadModelBtn');
const resetBtn = document.getElementById('resetBtn');
const toggleVisorBtn = document.getElementById('toggleVisorBtn');
const modelJsonInput = document.getElementById('modelJsonInput');
const trainingLogs = document.getElementById('trainingLogs');
const modelInfo = document.getElementById('modelInfo');
const metricPrimary = document.getElementById('metricPrimary');
const metricSecondary = document.getElementById('metricSecondary');
const previewContainer = document.getElementById('previewContainer');
const noiseSlider = document.getElementById('noiseSlider');
const noiseValue = document.getElementById('noiseValue');
const modeRadios = document.querySelectorAll('input[name="mode"]');

// helpers
function log(msg) { trainingLogs.innerHTML += '> ' + msg + '<br>'; trainingLogs.scrollTop = trainingLogs.scrollHeight; }
function setModelInfo(text) { modelInfo.innerText = '🧠 ' + text; }

// dispose all tensors and models
function disposeAll() {
    if (trainXs) { tf.dispose(trainXs); trainXs = null; }
    if (trainYs) { tf.dispose(trainYs); trainYs = null; }
    if (testXs) { tf.dispose(testXs); testXs = null; }
    if (testYs) { tf.dispose(testYs); testYs = null; }
    if (classificationModel) { classificationModel.dispose(); classificationModel = null; }
    if (modelMax) { modelMax.dispose(); modelMax = null; }
    if (modelAvg) { modelAvg.dispose(); modelAvg = null; }
    setModelInfo('none (disposed)');
    metricPrimary.innerText = '-';
    metricSecondary.innerText = '-';
}

// reset UI
resetBtn.addEventListener('click', () => {
    disposeAll();
    previewContainer.innerHTML = '<div style="color:#8895aa; text-align:center; padding:20px;">↑ upload data and click "Test 5 random" ↑</div>';
    dataStatus.innerText = 'no data';
    log('🧹 reset complete');
});

// mode switching
modeRadios.forEach(r => r.addEventListener('change', (e) => {
    currentMode = e.target.value;
    log(`mode switched to ${currentMode}`);
}));
noiseSlider.addEventListener('input', (e) => {
    noiseLevel = parseFloat(e.target.value);
    noiseValue.innerText = noiseLevel.toFixed(2);
});

// Load Data from both files
loadDataBtn.addEventListener('click', async () => {
    if (!trainFile.files[0] || !testFile.files[0]) {
        alert('please select both train and test CSV files');
        return;
    }
    try {
        disposeAll(); // fresh start
        log('📖 loading train CSV...');
        const train = await dataLoader.loadTrainFromFiles(trainFile.files[0]);
        trainXs = train.xs; trainYs = train.ys;
        log(`train samples: ${trainXs.shape[0]}`);

        log('📖 loading test CSV...');
        const test = await dataLoader.loadTestFromFiles(testFile.files[0]);
        testXs = test.xs; testYs = test.ys;
        log(`test samples: ${testXs.shape[0]}`);

        dataStatus.innerText = `✅ train:${trainXs.shape[0]} test:${testXs.shape[0]}`;
    } catch (err) {
        console.error(err);
        alert('error loading files: ' + err.message);
    }
});

// ---------- model builders ----------
function createClassificationModel() {
    const model = tf.sequential({
        layers: [
            tf.layers.conv2d({ filters:32, kernelSize:3, activation:'relu', padding:'same', inputShape:[28,28,1] }),
            tf.layers.conv2d({ filters:64, kernelSize:3, activation:'relu', padding:'same' }),
            tf.layers.maxPooling2d({ poolSize:2 }),
            tf.layers.dropout({ rate:0.25 }),
            tf.layers.flatten(),
            tf.layers.dense({ units:128, activation:'relu' }),
            tf.layers.dropout({ rate:0.5 }),
            tf.layers.dense({ units:10, activation:'softmax' })
        ]
    });
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
    return model;
}

function createDenoisingAutoencoder(poolingType = 'max') {
    const poolLayer = poolingType === 'max' ? tf.layers.maxPooling2d : tf.layers.averagePooling2d;
    const upsampleLayer = tf.layers.upSampling2d; // same for both

    const model = tf.sequential();
    // encoder
    model.add(tf.layers.conv2d({ filters:32, kernelSize:3, activation:'relu', padding:'same', inputShape:[28,28,1] }));
    model.add(poolLayer({ poolSize:2 }));
    model.add(tf.layers.conv2d({ filters:64, kernelSize:3, activation:'relu', padding:'same' }));
    model.add(poolLayer({ poolSize:2 }));
    // decoder
    model.add(tf.layers.conv2d({ filters:64, kernelSize:3, activation:'relu', padding:'same' }));
    model.add(upsampleLayer({ size:2 }));
    model.add(tf.layers.conv2d({ filters:32, kernelSize:3, activation:'relu', padding:'same' }));
    model.add(upsampleLayer({ size:2 }));
    model.add(tf.layers.conv2d({ filters:1, kernelSize:3, activation:'sigmoid', padding:'same' }));

    model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });
    return model;
}

// ---------- training ----------
trainBtn.addEventListener('click', async () => {
    if (!trainXs) { alert('load training data first'); return; }
    try {
        log(`🏋️ starting training in mode: ${currentMode} ...`);

        if (currentMode === 'classification') {
            if (!classificationModel) classificationModel = createClassificationModel();
            const { trainXs: xTrain, trainYs: yTrain, valXs: xVal, valYs: yVal } = dataLoader.splitTrainVal(trainXs, trainYs, 0.1);

            const callbacks = tfvis.show.fitCallbacks(
                { name: 'classification training' },
                ['loss', 'val_loss', 'acc', 'val_acc'],
                { callbacks: ['onEpochEnd'], zoomToFit: true }
            );

            await classificationModel.fit(xTrain, yTrain, {
                epochs: 5,
                batchSize: 128,
                validationData: [xVal, yVal],
                callbacks: callbacks,
                shuffle: true
            });

            tf.dispose([xTrain, yTrain, xVal, yVal]);
            setModelInfo('classification trained');
            log('✅ classification done');

        } else { // denoising modes
            const isMax = currentMode === 'denoising-max';
            let model = isMax ? modelMax : modelAvg;
            if (!model) {
                model = createDenoisingAutoencoder(isMax ? 'max' : 'avg');
                if (isMax) modelMax = model; else modelAvg = model;
            }

            // For denoising, we use trainXs as both input (noisy) and target (clean)
            // We create noisy version on fly inside training loop via data augmentation?
            // simpler: create static noisy version each epoch? we can use .fit with generator?
            // but for simplicity: precompute noisy once? but then overfit. Let's do on-the-fly with tf.data.generator

            log('preparing denoising dataset with on-the-fly noise...');
            const numSamples = trainXs.shape[0];
            const batchSize = 64;

            // create generator that yields {xs: noisy, ys: clean}
            function* dataGenerator() {
                const indices = tf.util.createShuffledIndices(numSamples);
                for (let i = 0; i < numSamples; i += batchSize) {
                    const batchIndices = indices.slice(i, i + batchSize);
                    const cleanBatch = tf.gather(trainXs, batchIndices);
                    const noisyBatch = dataLoader.addRandomNoise(cleanBatch, noiseLevel);
                    yield { xs: noisyBatch, ys: cleanBatch };
                    tf.dispose([cleanBatch, noisyBatch]);
                }
            }

            const ds = tf.data.generator(dataGenerator).repeat();
            // We'll use fitDataset for simplicity and show loss via vis callback
            const callbacks = tfvis.show.fitCallbacks(
                { name: `denoising ${isMax ? 'MaxPool' : 'AvgPool'}` },
                ['loss'],
                { callbacks: ['onEpochEnd'] }
            );

            await model.fitDataset(ds, {
                epochs: 6,
                batchesPerEpoch: Math.ceil(numSamples / batchSize),
                callbacks: callbacks,
                validationData: async () => {
                    // small validation batch
                    const { xs: valBatch, ys: valLabels } = dataLoader.getRandomTestBatch(testXs, testYs, 200);
                    const valNoisy = dataLoader.addRandomNoise(valBatch, noiseLevel);
                    return { xs: valNoisy, ys: valBatch };
                }
            });

            setModelInfo(`denoiser ${isMax ? 'Max' : 'Avg'} ready`);
            log(`✅ denoising model (${isMax ? 'MaxPool' : 'AvgPool'}) trained`);
        }
    } catch (err) {
        console.error(err);
        alert('training error: ' + err.message);
    }
});

// ---------- evaluate ----------
evaluateBtn.addEventListener('click', async () => {
    if (!testXs) { alert('load test data first'); return; }
    if (currentMode === 'classification') {
        if (!classificationModel) { alert('no classification model'); return; }
        const acc = classificationModel.evaluate(testXs, testYs)[1].dataSync()[0];
        metricPrimary.innerText = (acc*100).toFixed(1) + '%';
        // confusion matrix & per-class via tfvis
        const preds = classificationModel.predict(testXs);
        const trueLabels = testYs.argMax(-1);
        const predLabels = preds.argMax(-1);
        const confusionMatrix = await tfvis.metrics.confusionMatrix(trueLabels, predLabels, 10);
        tfvis.render.confusionMatrix({ name: 'Confusion Matrix' }, confusionMatrix);
        tf.dispose(preds);
        log(`evaluation accuracy: ${(acc*100).toFixed(2)}%`);
    } else {
        // denoising: compute MSE on noisy vs reconstructed
        const isMax = currentMode === 'denoising-max';
        const model = isMax ? modelMax : modelAvg;
        if (!model) { alert('train denoising model first'); return; }

        const { xs: cleanBatch, ys: _ } = dataLoader.getRandomTestBatch(testXs, testYs, 200);
        const noisyBatch = dataLoader.addRandomNoise(cleanBatch, noiseLevel);
        const reconstructed = model.predict(noisyBatch);
        const mse = await dataLoader.calculateMSE(cleanBatch, reconstructed);
        metricPrimary.innerText = mse.toFixed(4);
        tf.dispose([cleanBatch, noisyBatch, reconstructed]);
        log(`MSE on noisy test: ${mse.toFixed(6)}`);
    }
});

// ---------- Test 5 random (core homework step 3) ----------
testFiveBtn.addEventListener('click', async () => {
    if (!testXs) { alert('load test data first'); return; }
    const k = 5;
    const { xs: clean5, ys: label5 } = dataLoader.getRandomTestBatch(testXs, testYs, k);
    const noisy5 = dataLoader.addRandomNoise(clean5, noiseLevel);

    // prepare containers
    previewContainer.innerHTML = '';
    const hasMax = modelMax !== null;
    const hasAvg = modelAvg !== null;

    for (let i = 0; i < k; i++) {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'sample-row';

        // helper to add a canvas block
        const addCanvas = async (tensor, label, mseVal = null) => {
            const block = document.createElement('div');
            block.className = 'canvas-block';
            const canvas = document.createElement('canvas');
            await dataLoader.draw28x28ToCanvas(tensor.slice([i,0,0,0], [1,28,28,1]), canvas, 2);
            block.appendChild(canvas);
            const lbl = document.createElement('div');
            lbl.className = 'canvas-label';
            lbl.innerText = label;
            block.appendChild(lbl);
            if (mseVal !== null) {
                const mseSpan = document.createElement('span');
                mseSpan.className = 'mse-badge';
                mseSpan.innerText = `MSE:${mseVal.toFixed(3)}`;
                block.appendChild(mseSpan);
            }
            rowDiv.appendChild(block);
        };

        // original
        await addCanvas(clean5, 'orig');

        // noisy
        await addCanvas(noisy5, `noisy(${noiseLevel})`);

        // MaxPool denoised if exists
        if (modelMax) {
            const recMax = modelMax.predict(noisy5.slice([i,0,0,0], [1,28,28,1]));
            const mseMax = await dataLoader.calculateMSE(clean5.slice([i,0,0,0], [1,28,28,1]), recMax);
            await addCanvas(recMax, 'MaxPool', mseMax);
            tf.dispose(recMax);
        } else {
            const empty = document.createElement('div'); empty.innerText = 'no Max'; rowDiv.appendChild(empty);
        }

        // AvgPool denoised if exists
        if (modelAvg) {
            const recAvg = modelAvg.predict(noisy5.slice([i,0,0,0], [1,28,28,1]));
            const mseAvg = await dataLoader.calculateMSE(clean5.slice([i,0,0,0], [1,28,28,1]), recAvg);
            await addCanvas(recAvg, 'AvgPool', mseAvg);
            tf.dispose(recAvg);
        } else {
            const empty = document.createElement('div'); empty.innerText = 'no Avg'; rowDiv.appendChild(empty);
        }

        previewContainer.appendChild(rowDiv);
    }

    tf.dispose([clean5, noisy5, label5]);
    log('preview updated');
});

// ---------- Compare pooling (step 3 extra) ----------
comparePoolingBtn.addEventListener('click', async () => {
    if (!trainXs) { alert('load data first'); return; }
    log('🔄 training both autoencoders sequentially (Max then Avg)...');

    // Train MaxPool if not exists
    if (!modelMax) modelMax = createDenoisingAutoencoder('max');
    currentMode = 'denoising-max';
    await trainBtn.click();   // quick hack: simulate click (but we need proper await)
    // but we want to ensure both finish: better call internal train function? 
    // For clarity, we re-use but must wait. We'll make a local async train function.

    async function trainOne(modelType) {
        const isMax = modelType === 'max';
        const model = isMax ? (modelMax || createDenoisingAutoencoder('max')) : (modelAvg || createDenoisingAutoencoder('avg'));
        if (isMax) modelMax = model; else modelAvg = model;

        const numSamples = trainXs.shape[0];
        const batchSize = 64;
        function* gen() {
            const indices = tf.util.createShuffledIndices(numSamples);
            for (let i = 0; i < numSamples; i += batchSize) {
                const batchIndices = indices.slice(i, i + batchSize);
                const cleanBatch = tf.gather(trainXs, batchIndices);
                const noisyBatch = dataLoader.addRandomNoise(cleanBatch, noiseLevel);
                yield { xs: noisyBatch, ys: cleanBatch };
                tf.dispose([cleanBatch, noisyBatch]);
            }
        }
        const ds = tf.data.generator(gen).repeat();
        await model.fitDataset(ds, {
            epochs: 4,
            batchesPerEpoch: Math.ceil(numSamples / batchSize),
            callbacks: tfvis.show.fitCallbacks({ name: `Compare ${modelType}` }, ['loss'])
        });
    }

    await trainOne('max');
    await trainOne('avg');
    log('✅ both models trained – showing comparison