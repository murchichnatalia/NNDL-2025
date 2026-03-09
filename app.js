app.js
```javascript
class MNISTApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();
        this.classificationModel = null;
        this.modelMax = null;
        this.modelAvg = null;
        this.isTraining = false;
        this.trainData = null;
        this.testData = null;
        this.currentMode = 'classification';
        this.noiseLevel = 0.3;
        
        this.initializeUI();
    }

    initializeUI() {
        // Bind button events - используем правильные ID из HTML
        document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
        document.getElementById('trainBtn').addEventListener('click', () => this.onTrain());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.onEvaluate());
        document.getElementById('testFiveBtn').addEventListener('click', () => this.onTestFive());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.onSaveDownload()); // saveModelBtn, не saveBtn
        document.getElementById('loadModelBtn').addEventListener('click', () => this.onLoadFromFiles());
        document.getElementById('resetBtn').addEventListener('click', () => this.onReset());
        document.getElementById('toggleVisorBtn').addEventListener('click', () => this.toggleVisor());
        
        // Кнопка сравнения (если есть в HTML)
        const compareBtn = document.getElementById('comparePoolingBtn');
        if (compareBtn) {
            compareBtn.addEventListener('click', () => this.onComparePooling());
        }
        
        // Noise slider
        const noiseSlider = document.getElementById('noiseSlider');
        if (noiseSlider) {
            noiseSlider.addEventListener('input', (e) => {
                this.noiseLevel = parseFloat(e.target.value);
                const noiseValue = document.getElementById('noiseValue');
                if (noiseValue) noiseValue.textContent = this.noiseLevel.toFixed(2);
            });
        }
        
        // Mode radios
        const modeRadios = document.querySelectorAll('input[name="mode"]');
        modeRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentMode = e.target.value;
                this.showStatus(`Switched to ${this.currentMode} mode`);
                this.updateModelInfo();
            });
        });
    }

    async onLoadData() {
        try {
            const trainFile = document.getElementById('trainFile').files[0];
            const testFile = document.getElementById('testFile').files[0];
            
            if (!trainFile || !testFile) {
                this.showError('Please select both train and test CSV files');
                return;
            }

            this.showStatus('Loading training data...');
            const trainData = await this.dataLoader.loadTrainFromFiles(trainFile);
            
            this.showStatus('Loading test data...');
            const testData = await this.dataLoader.loadTestFromFiles(testFile);

            this.trainData = trainData;
            this.testData = testData;

            this.updateDataStatus(trainData.xs.shape[0], testData.xs.shape[0]);
            this.showStatus('Data loaded successfully!');
            
        } catch (error) {
            this.showError(`Failed to load data: ${error.message}`);
        }
    }

    async onTrain() {
        if (!this.trainData) {
            this.showError('Please load training data first');
            return;
        }

        if (this.isTraining) {
            this.showError('Training already in progress');
            return;
        }

        try {
            this.isTraining = true;
            
            if (this.currentMode === 'classification') {
                await this.trainClassificationModel();
            } else {
                await this.trainDenoisingModel();
            }
            
        } catch (error) {
            this.showError(`Training failed: ${error.message}`);
        } finally {
            this.isTraining = false;
        }
    }

    async trainClassificationModel() {
        this.showStatus('Starting classification training...');
        
        const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
            this.trainData.xs, this.trainData.ys, 0.1
        );

        if (!this.classificationModel) {
            this.classificationModel = this.createClassificationModel();
        }

        const startTime = Date.now();
        const history = await this.classificationModel.fit(trainXs, trainYs, {
            epochs: 5,
            batchSize: 128,
            validationData: [valXs, valYs],
            shuffle: true,
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Classification Training' },
                ['loss', 'val_loss', 'acc', 'val_acc'],
                { callbacks: ['onEpochEnd'] }
            )
        });

        const duration = (Date.now() - startTime) / 1000;
        const bestValAcc = Math.max(...history.history.val_acc);
        
        this.showStatus(`Classification training completed in ${duration.toFixed(1)}s. Best val_acc: ${bestValAcc.toFixed(4)}`);
        
        trainXs.dispose();
        trainYs.dispose();
        valXs.dispose();
        valYs.dispose();
        
        this.updateModelInfo();
    }

    async trainDenoisingModel() {
        const isMax = this.currentMode === 'denoising-max';
        const poolingType = isMax ? 'max' : 'avg';
        
        this.showStatus(`Starting denoising autoencoder training (${poolingType} pooling)...`);
        
        let model = isMax ? this.modelMax : this.modelAvg;
        if (!model) {
            model = this.createDenoisingAutoencoder(poolingType);
            if (isMax) {
                this.modelMax = model;
            } else {
                this.modelAvg = model;
            }
        }

        const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
            this.trainData.xs, this.trainData.ys, 0.1
        );

        const startTime = Date.now();
        
        const history = await model.fit(trainXs, trainXs, {
            epochs: 8,
            batchSize: 64,
            validationData: [valXs, valXs],
            shuffle: true,
            callbacks: tfvis.show.fitCallbacks(
                { name: `Denoising ${poolingType} Training` },
                ['loss', 'val_loss'],
                { callbacks: ['onEpochEnd'] }
            )
        });

        const duration = (Date.now() - startTime) / 1000;
        const bestValLoss = Math.min(...history.history.val_loss);
        
        this.showStatus(`Denoising autoencoder (${poolingType}) trained in ${duration.toFixed(1)}s. Best val_loss: ${bestValLoss.toFixed(4)}`);
        
        trainXs.dispose();
        trainYs.dispose();
        valXs.dispose();
        valYs.dispose();
        
        this.updateModelInfo();
    }

    createClassificationModel() {
        const model = tf.sequential();
        
        model.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same',
            inputShape: [28, 28, 1]
        }));
        
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        
        model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
        model.add(tf.layers.dropout({ rate: 0.25 }));
        model.add(tf.layers.flatten());
        
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        model.add(tf.layers.dropout({ rate: 0.5 }));
        model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
        
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        return model;
    }

    createDenoisingAutoencoder(poolingType = 'max') {
        const model = tf.sequential();
        
        const poolLayer = poolingType === 'max' ? 
            tf.layers.maxPooling2d({ poolSize: 2 }) : 
            tf.layers.averagePooling2d({ poolSize: 2 });
        
        // Encoder
        model.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same',
            inputShape: [28, 28, 1]
        }));
        model.add(poolLayer);
        
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        model.add(poolLayer);
        
        // Decoder
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        model.add(tf.layers.upSampling2d({ size: 2 }));
        
        model.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        model.add(tf.layers.upSampling2d({ size: 2 }));
        
        model.add(tf.layers.conv2d({
            filters: 1,
            kernelSize: 3,
            activation: 'sigmoid',
            padding: 'same'
        }));
        
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });
        
        return model;
    }

    async onTestFive() {
        if (!this.testData) {
            this.showError('Please load test data first');
            return;
        }

        try {
            const { xs: cleanBatch, ys: labelBatch } = this.dataLoader.getRandomTestBatch(
                this.testData.xs, this.testData.ys, 5
            );
            
            const noisyBatch = this.dataLoader.addRandomNoise(cleanBatch, this.noiseLevel);
            
            let reconstructionsMax = null;
            let reconstructionsAvg = null;
            
            if (this.modelMax) {
                reconstructionsMax = this.modelMax.predict(noisyBatch);
            }
            
            if (this.modelAvg) {
                reconstructionsAvg = this.modelAvg.predict(noisyBatch);
            }
            
            await this.renderPreview(cleanBatch, noisyBatch, reconstructionsMax, reconstructionsAvg);
            
            cleanBatch.dispose();
            noisyBatch.dispose();
            if (reconstructionsMax) reconstructionsMax.dispose();
            if (reconstructionsAvg) reconstructionsAvg.dispose();
            
        } catch (error) {
            this.showError(`Test preview failed: ${error.message}`);
        }
    }

    async renderPreview(cleanBatch, noisyBatch, recMax, recAvg) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '';
        
        const previewRow = document.createElement('div');
        previewRow.className = 'preview-row';
        
        const cleanArray = await cleanBatch.array();
        const noisyArray = await noisyBatch.array();
        
        for (let i = 0; i < 5; i++) {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'preview-item';
            
            // Original
            const origCanvas = document.createElement('canvas');
            const origTensor = cleanBatch.slice([i,0,0,0], [1,28,28,1]);
            await this.dataLoader.draw28x28ToCanvas(origTensor, origCanvas, 2);
            origTensor.dispose();
            
            // Noisy
            const noisyCanvas = document.createElement('canvas');
            const noisyTensor = noisyBatch.slice([i,0,0,0], [1,28,28,1]);
            await this.dataLoader.draw28x28ToCanvas(noisyTensor, noisyCanvas, 2);
            noisyTensor.dispose();
            
            itemDiv.appendChild(origCanvas);
            itemDiv.appendChild(noisyCanvas);
            
            // Add reconstructions if available
            if (recMax) {
                const maxCanvas = document.createElement('canvas');
                const maxTensor = recMax.slice([i,0,0,0], [1,28,28,1]);
                await this.dataLoader.draw28x28ToCanvas(maxTensor, maxCanvas, 2);
                maxTensor.dispose();
                itemDiv.appendChild(maxCanvas);
            }
            
            if (recAvg) {
                const avgCanvas = document.createElement('canvas');
                const avgTensor = recAvg.slice([i,0,0,0], [1,28,28,1]);
                await this.dataLoader.draw28x28ToCanvas(avgTensor, avgCanvas, 2);
                avgTensor.dispose();
                itemDiv.appendChild(avgCanvas);
            }
            
            previewRow.appendChild(itemDiv);
        }
        
        container.appendChild(previewRow);
    }

    async onSaveDownload() {
        let modelToSave = null;
        let filename = '';
        
        if (this.currentMode === 'classification' && this.classificationModel) {
            modelToSave = this.classificationModel;
            filename = 'mnist-classifier';
        } else if (this.currentMode === 'denoising-max' && this.modelMax) {
            modelToSave = this.modelMax;
            filename = 'mnist-denoiser-max';
        } else if (this.currentMode === 'denoising-avg' && this.modelAvg) {
            modelToSave = this.modelAvg;
            filename = 'mnist-denoiser-avg';
        } else {
            this.showError(`No ${this.currentMode} model available to save`);
            return;
        }

        try {
            await modelToSave.save(`downloads://${filename}`);
            this.showStatus(`Model saved as ${filename}`);
        } catch (error) {
            this.showError(`Failed to save model: ${error.message}`);
        }
    }

    async onLoadFromFiles() {
        const jsonFile = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];
        
        if (!jsonFile || !weightsFile) {
            this.showError('Please select both model.json and weights.bin files');
            return;
        }

        try {
            this.showStatus('Loading model...');
            
            const loadedModel = await tf.loadLayersModel(
                tf.io.browserFiles([jsonFile, weightsFile])
            );
            
            // Simple detection: check output shape
            const outputShape = loadedModel.outputs[0].shape;
            if (outputShape[outputShape.length - 1] === 10) {
                // Classification model
                if (this.classificationModel) this.classificationModel.dispose();
                this.classificationModel = loadedModel;
                this.currentMode = 'classification';
                this.showStatus('Loaded classification model');
            } else {
                // Denoising model - assume MaxPool by default
                if (this.modelMax) this.modelMax.dispose();
                this.modelMax = loadedModel;
                this.currentMode = 'denoising-max';
                this.showStatus('Loaded denoising autoencoder');
            }
            
            // Update radio button
            const radio = document.querySelector(`input[name="mode"][value="${this.currentMode}"]`);
            if (radio) radio.checked = true;
            
            this.updateModelInfo();
            
        } catch (error) {
            this.showError(`Failed to load model: ${error.message}`);
        }
    }

    async onEvaluate() {
        if (!this.testData) {
            this.showError('No test data available');
            return;
        }

        this.showStatus('Evaluation not fully implemented in this version');
    }

    async onComparePooling() {
        if (!this.trainData) {
            this.showError('Please load training data first');
            return;
        }

        this.showStatus('Starting pooling comparison training...');
        
        // Train MaxPool
        this.currentMode = 'denoising-max';
        await this.trainDenoisingModel();
        
        // Train AvgPool
        this.currentMode = 'denoising-avg';
        await this.trainDenoisingModel();
        
        // Show comparison
        await this.onTestFive();
        
        this.showStatus('Pooling comparison completed!');
    }

    onReset() {
        if (this.classificationModel) {
            this.classificationModel.dispose();
            this.classificationModel = null;
        }
        if (this.modelMax) {
            this.modelMax.dispose();
            this.modelMax = null;
        }
        if (this.modelAvg) {
            this.modelAvg.dispose();
            this.modelAvg = null;
        }
        
        if (this.dataLoader && this.dataLoader.dispose) {
            this.dataLoader.dispose();
        }
        
        this.trainData = null;
        this.testData = null;
        
        this.updateDataStatus(0, 0);
        this.updateModelInfo();
        this.clearPreview();
        this.showStatus('Reset completed');
        
        const metricPrimary = document.getElementById('metricPrimary');
        const metricSecondary = document.getElementById('metricSecondary');
        if (metricPrimary) metricPrimary.textContent = '-';
        if (metricSecondary) metricSecondary.textContent = '-';
    }

    toggleVisor() {
        tfvis.visor().toggle();
    }

    clearPreview() {
        const container = document.getElementById('previewContainer');
        if (container) {
            container.innerHTML = '<p style="color: #999; text-align: center;">Click "Test 5 Random" to see results</p>';
        }
    }

    updateDataStatus(trainCount, testCount) {
        const statusEl = document.getElementById('dataStatus');
        if (statusEl) {
            statusEl.innerHTML = `
                <h3>Data Status</h3>
                <p>Train samples: ${trainCount}</p>
                <p>Test samples: ${testCount}</p>
            `;
        }
    }

    updateModelInfo() {
        const infoEl = document.getElementById('modelInfo');
        if (!infoEl) return;
        
        let model = null;
        let modelType = 'None';
        
        if (this.currentMode === 'classification' && this.classificationModel) {
            model = this.classificationModel;
            modelType = 'Classification CNN';
        } else if (this.currentMode === 'denoising-max' && this.modelMax) {
            model = this.modelMax;
            modelType = 'Denoising Autoencoder (MaxPool)';
        } else if (this.currentMode === 'denoising-avg' && this.modelAvg) {
            model = this.modelAvg;
            modelType = 'Denoising Autoencoder (AvgPool)';
        }
        
        if (!model) {
            infoEl.innerHTML = '<h3>Model Info</h3><p>No model loaded</p>';
            return;
        }
        
        let totalParams = 0;
        model.layers.forEach(layer => {
            layer.getWeights().forEach(weight => {
                totalParams += weight.size;
            });
        });
        
        infoEl.innerHTML = `
            <h3>Model Info</h3>
            <p><strong>${modelType}</strong></p>
            <p>Layers: ${model.layers.length}</p>
            <p>Total parameters: ${totalParams.toLocaleString()}</p>
            <p>Input shape: [28,28,1]</p>
        `;
    }

    showStatus(message) {
        const logs = document.getElementById('trainingLogs');
        if (logs) {
            const entry = document.createElement('div');
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logs.appendChild(entry);
            logs.scrollTop = logs.scrollHeight;
        }
        console.log(message);
    }

    showError(message) {
        this.showStatus(`ERROR: ${message}`);
        console.error(message);
    }
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', () => {
    new MNISTApp();
});
