app.js
```javascript
// app.js - Modified for Denoising Autoencoder Homework (4 steps)
// Includes: noise addition, autoencoder training, pooling comparison, save/load

class MNISTApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();
        // Three separate models for classification and denoising comparison
        this.classificationModel = null;  // Original CNN for digits
        this.modelMax = null;             // Autoencoder with MaxPooling
        this.modelAvg = null;              // Autoencoder with AveragePooling
        this.isTraining = false;
        this.trainData = null;
        this.testData = null;
        
        // New state variables for homework
        this.currentMode = 'classification'; // 'classification' | 'denoising-max' | 'denoising-avg'
        this.noiseLevel = 0.3;               // Default noise level
        
        this.initializeUI();
        this.setupModeSwitching();
    }

    initializeUI() {
        // Bind button events
        document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
        document.getElementById('trainBtn').addEventListener('click', () => this.onTrain());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.onEvaluate());
        document.getElementById('testFiveBtn').addEventListener('click', () => this.onTestFive());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.onSaveDownload());
        document.getElementById('loadModelBtn').addEventListener('click', () => this.onLoadFromFiles());
        document.getElementById('resetBtn').addEventListener('click', () => this.onReset());
        document.getElementById('toggleVisorBtn').addEventListener('click', () => this.toggleVisor());
        document.getElementById('comparePoolingBtn').addEventListener('click', () => this.onComparePooling());
        
        // New: Noise slider
        document.getElementById('noiseSlider').addEventListener('input', (e) => {
            this.noiseLevel = parseFloat(e.target.value);
            document.getElementById('noiseValue').textContent = this.noiseLevel.toFixed(2);
        });
    }

    setupModeSwitching() {
        const modeRadios = document.querySelectorAll('input[name="mode"]');
        modeRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentMode = e.target.value;
                this.showStatus(`Switched to ${this.currentMode} mode`);
                this.updateModelInfo(); // Update display for current model
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

    // Step 2: Train CNN autoencoder for denoising
    async trainDenoisingModel() {
        const isMax = this.currentMode === 'denoising-max';
        const poolingType = isMax ? 'max' : 'avg';
        
        this.showStatus(`Starting denoising autoencoder training (${poolingType} pooling)...`);
        
        // Get or create model
        let model = isMax ? this.modelMax : this.modelAvg;
        if (!model) {
            model = this.createDenoisingAutoencoder(poolingType);
            if (isMax) {
                this.modelMax = model;
            } else {
                this.modelAvg = model;
            }
        }

        // Split training data
        const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
            this.trainData.xs, this.trainData.ys, 0.1
        );

        // Step 1: Add noise to training data on-the-fly
        const startTime = Date.now();
        
        // Create custom callback for denoising visualization
        const callbacks = {
            onEpochEnd: async (epoch, logs) => {
                // Show sample reconstructions every epoch
                if (epoch % 2 === 0) {
                    await this.visualizeDenoisingProgress(model, epoch);
                }
                
                // Update loss display
                tfvis.show.fitCallbacks(
                    { name: `Denoising ${poolingType} Training` },
                    ['loss', 'val_loss'],
                    { callbacks: ['onEpochEnd'] }
                ).onEpochEnd(epoch, logs);
                
                this.showStatus(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, val_loss = ${logs.val_loss.toFixed(4)}`);
            }
        };

        // Train denoising autoencoder
        const history = await model.fit(trainXs, trainXs, {  // Autoencoder: input = target
            epochs: 8,  // Denoising needs more epochs
            batchSize: 64,
            validationData: [valXs, valXs],
            shuffle: true,
            callbacks: callbacks
        });

        const duration = (Date.now() - startTime) / 1000;
        const bestValLoss = Math.min(...history.history.val_loss);
        
        this.showStatus(`Denoising autoencoder (${poolingType}) trained in ${duration.toFixed(1)}s. Best val_loss: ${bestValLoss.toFixed(4)}`);
        
        // Clean up
        trainXs.dispose();
        trainYs.dispose();
        valXs.dispose();
        valYs.dispose();
        
        this.updateModelInfo();
    }

    async trainClassificationModel() {
        this.showStatus('Starting classification training...');
        
        // Split training data
        const { trainXs, trainYs, valXs, valYs } = this.dataLoader.splitTrainVal(
            this.trainData.xs, this.trainData.ys, 0.1
        );

        // Create or get model
        if (!this.classificationModel) {
            this.classificationModel = this.createClassificationModel();
        }

        // Train with tfjs-vis callbacks
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
        
        // Clean up
        trainXs.dispose();
        trainYs.dispose();
        valXs.dispose();
        valYs.dispose();
        
        this.updateModelInfo();
    }

    // Step 2: Create denoising autoencoder with specified pooling type
    createDenoisingAutoencoder(poolingType = 'max') {
        const model = tf.sequential();
        
        // Choose pooling layer based on type
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
            loss: 'meanSquaredError'  // MSE for image reconstruction
        });
        
        return model;
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

    async onEvaluate() {
        if (this.currentMode === 'classification') {
            await this.evaluateClassification();
        } else {
            await this.evaluateDenoising();
        }
    }

    async evaluateClassification() {
        if (!this.classificationModel) {
            this.showError('No classification model available');
            return;
        }

        if (!this.testData) {
            this.showError('No test data available');
            return;
        }

        try {
            this.showStatus('Evaluating classification model...');
            
            const testXs = this.testData.xs;
            const testYs = this.testData.ys;
            
            // Get predictions
            const predictions = this.classificationModel.predict(testXs);
            const predictedLabels = predictions.argMax(-1);
            const trueLabels = testYs.argMax(-1);
            
            // Calculate accuracy
            const accuracy = await this.calculateAccuracy(predictedLabels, trueLabels);
            
            // Create confusion matrix
            const confusionMatrix = await this.createConfusionMatrix(predictedLabels, trueLabels);
            
            // Show metrics in visor
            const metricsContainer = { name: 'Classification Metrics', tab: 'Evaluation' };
            
            tfvis.show.perClassAccuracy(metricsContainer, 
                this.calculatePerClassAccuracy(confusionMatrix), 
                ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            );
            
            tfvis.render.confusionMatrix(metricsContainer, {
                values: confusionMatrix,
                tickLabels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            });
            
            this.showStatus(`Test accuracy: ${(accuracy * 100).toFixed(2)}%`);
            document.getElementById('metricPrimary').textContent = `${(accuracy * 100).toFixed(2)}%`;
            
            // Clean up
            predictions.dispose();
            predictedLabels.dispose();
            trueLabels.dispose();
            
        } catch (error) {
            this.showError(`Evaluation failed: ${error.message}`);
        }
    }

    async evaluateDenoising() {
        if (!this.modelMax && !this.modelAvg) {
            this.showError('No denoising models available');
            return;
        }

        try {
            this.showStatus('Evaluating denoising models...');
            
            // Get a batch of test images
            const { xs: cleanBatch, ys: _ } = this.dataLoader.getRandomTestBatch(
                this.testData.xs, this.testData.ys, 100
            );
            
            // Step 1: Add noise
            const noisyBatch = this.dataLoader.addRandomNoise(cleanBatch, this.noiseLevel);
            
            const results = [];
            
            // Evaluate both models if they exist
            if (this.modelMax) {
                const reconstructed = this.modelMax.predict(noisyBatch);
                const mse = await this.dataLoader.calculateMSE(cleanBatch, reconstructed);
                results.push({ type: 'MaxPool', mse });
                reconstructed.dispose();
            }
            
            if (this.modelAvg) {
                const reconstructed = this.modelAvg.predict(noisyBatch);
                const mse = await this.dataLoader.calculateMSE(cleanBatch, reconstructed);
                results.push({ type: 'AvgPool', mse });
                reconstructed.dispose();
            }
            
            // Display results
            let resultText = 'MSE results: ';
            results.forEach(r => resultText += `${r.type}: ${r.mse.toFixed(6)} `);
            this.showStatus(resultText);
            
            if (results.length > 0) {
                document.getElementById('metricPrimary').textContent = results[0].mse.toFixed(6);
                if (results.length > 1) {
                    document.getElementById('metricSecondary').textContent = results[1].mse.toFixed(6);
                }
            }
            
            // Clean up
            cleanBatch.dispose();
            noisyBatch.dispose();
            
        } catch (error) {
            this.showError(`Denoising evaluation failed: ${error.message}`);
        }
    }

    // Step 3: Modified Test 5 Random to show denoising comparison
    async onTestFive() {
        if (!this.testData) {
            this.showError('Please load test data first');
            return;
        }

        try {
            // Get 5 random test images
            const { xs: cleanBatch, ys: labelBatch, indices } = this.dataLoader.getRandomTestBatch(
                this.testData.xs, this.testData.ys, 5
            );
            
            // Step 1: Add noise
            const noisyBatch = this.dataLoader.addRandomNoise(cleanBatch, this.noiseLevel);
            
            // Get predictions from available models
            let reconstructionsMax = null;
            let reconstructionsAvg = null;
            
            if (this.modelMax) {
                reconstructionsMax = this.modelMax.predict(noisyBatch);
            }
            
            if (this.modelAvg) {
                reconstructionsAvg = this.modelAvg.predict(noisyBatch);
            }
            
            // Calculate MSE for each reconstruction
            const mseValues = [];
            for (let i = 0; i < 5; i++) {
                const mseEntry = { max: null, avg: null };
                if (reconstructionsMax) {
                    const cleanSlice = cleanBatch.slice([i,0,0,0], [1,28,28,1]);
                    const reconSlice = reconstructionsMax.slice([i,0,0,0], [1,28,28,1]);
                    mseEntry.max = await this.dataLoader.calculateMSE(cleanSlice, reconSlice);
                    cleanSlice.dispose();
                    reconSlice.dispose();
                }
                if (reconstructionsAvg) {
                    const cleanSlice = cleanBatch.slice([i,0,0,0], [1,28,28,1]);
                    const reconSlice = reconstructionsAvg.slice([i,0,0,0], [1,28,28,1]);
                    mseEntry.avg = await this.dataLoader.calculateMSE(cleanSlice, reconSlice);
                    cleanSlice.dispose();
                    reconSlice.dispose();
                }
                mseValues.push(mseEntry);
            }
            
            // Render comparison
            await this.renderDenoisingComparison(
                cleanBatch, 
                noisyBatch, 
                reconstructionsMax, 
                reconstructionsAvg,
                mseValues,
                indices
            );
            
            // Clean up
            cleanBatch.dispose();
            noisyBatch.dispose();
            if (reconstructionsMax) reconstructionsMax.dispose();
            if (reconstructionsAvg) reconstructionsAvg.dispose();
            
        } catch (error) {
            this.showError(`Test preview failed: ${error.message}`);
        }
    }

    // Render 4-image row for each sample: Original | Noisy | MaxPool | AvgPool
    async renderDenoisingComparison(cleanBatch, noisyBatch, recMax, recAvg, mseValues, indices) {
        const container = document.getElementById('previewContainer');
        container.innerHTML = '';
        
        const cleanArray = await cleanBatch.array();
        const noisyArray = await noisyBatch.array();
        
        for (let i = 0; i < 5; i++) {
            const rowDiv = document.createElement('div');
            rowDiv.className = 'preview-row';
            rowDiv.style.display = 'flex';
            rowDiv.style.gap = '10px';
            rowDiv.style.marginBottom = '20px';
            rowDiv.style.alignItems = 'center';
            
            // Helper to create canvas block
            const createCanvasBlock = async (tensor, title, mse = null) => {
                const blockDiv = document.createElement('div');
                blockDiv.style.textAlign = 'center';
                
                const canvas = document.createElement('canvas');
                await this.dataLoader.draw28x28ToCanvas(tensor, canvas, 2);
                
                const label = document.createElement('div');
                label.style.fontSize = '12px';
                label.style.marginTop = '4px';
                label.textContent = title;
                
                blockDiv.appendChild(canvas);
                blockDiv.appendChild(label);
                
                if (mse !== null) {
                    const mseSpan = document.createElement('div');
                    mseSpan.style.fontSize = '10px';
                    mseSpan.style.color = '#666';
                    mseSpan.textContent = `MSE: ${mse.toFixed(4)}`;
                    blockDiv.appendChild(mseSpan);
                }
                
                return blockDiv;
            };
            
            // Original image
            const origTensor = cleanBatch.slice([i,0,0,0], [1,28,28,1]);
            rowDiv.appendChild(await createCanvasBlock(origTensor, 'Original'));
            origTensor.dispose();
            
            // Noisy image
            const noisyTensor = noisyBatch.slice([i,0,0,0], [1,28,28,1]);
            rowDiv.appendChild(await createCanvasBlock(noisyTensor, `Noisy (${this.noiseLevel})`));
            noisyTensor.dispose();
            
            // MaxPool denoised (if available)
            if (recMax) {
                const maxTensor = recMax.slice([i,0,0,0], [1,28,28,1]);
                rowDiv.appendChild(await createCanvasBlock(maxTensor, 'MaxPool', mseValues[i].max));
                maxTensor.dispose();
            } else {
                const emptyDiv = document.createElement('div');
                emptyDiv.textContent = 'No MaxPool model';
                rowDiv.appendChild(emptyDiv);
            }
            
            // AvgPool denoised (if available)
            if (recAvg) {
                const avgTensor = recAvg.slice([i,0,0,0], [1,28,28,1]);
                rowDiv.appendChild(await createCanvasBlock(avgTensor, 'AvgPool', mseValues[i].avg));
                avgTensor.dispose();
            } else {
                const emptyDiv = document.createElement('div');
                emptyDiv.textContent = 'No AvgPool model';
                rowDiv.appendChild(emptyDiv);
            }
            
            container.appendChild(rowDiv);
        }
    }

    // Step 3: Compare pooling button
    async onComparePooling() {
        if (!this.trainData) {
            this.showError('Please load training data first');
            return;
        }

        this.showStatus('Starting pooling comparison training...');
        
        // Train MaxPool model
        this.currentMode = 'denoising-max';
        await this.trainDenoisingModel();
        
        // Train AvgPool model
        this.currentMode = 'denoising-avg';
        await this.trainDenoisingModel();
        
        // Show comparison
        await this.onTestFive();
        
        this.showStatus('Pooling comparison completed!');
    }

    // Step 4: Save trained model with appropriate naming
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

    // Step 4: Load model and detect type
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
            
            // Detect model type from architecture
            const config = loadedModel.toJSON(null, false);
            const isClassifier = config.modelTopology.config.layers.some(layer => 
                layer.class_name === 'Dense' && layer.config.units === 10
            );
            
            if (isClassifier) {
                // Classification model
                if (this.classificationModel) this.classificationModel.dispose();
                this.classificationModel = loadedModel;
                this.currentMode = 'classification';
                this.showStatus('Loaded classification model');
            } else {
                // Denoising autoencoder - detect pooling type
                const hasMaxPool = config.modelTopology.config.layers.some(layer => 
                    layer.class_name === 'MaxPooling2D'
                );
                
                if (hasMaxPool) {
                    if (this.modelMax) this.modelMax.dispose();
                    this.modelMax = loadedModel;
                    this.currentMode = 'denoising-max';
                    this.showStatus('Loaded MaxPool denoising autoencoder');
                } else {
                    if (this.modelAvg) this.modelAvg.dispose();
                    this.modelAvg = loadedModel;
                    this.currentMode = 'denoising-avg';
                    this.showStatus('Loaded AvgPool denoising autoencoder');
                }
            }
            
            // Update UI to reflect loaded mode
            document.querySelector(`input[name="mode"][value="${this.currentMode}"]`).checked = true;
            
            this.updateModelInfo();
            
        } catch (error) {
            this.showError(`Failed to load model: ${error.message}`);
        }
    }

    async visualizeDenoisingProgress(model, epoch) {
        if (!this.testData) return;
        
        // Get a single test image
        const { xs: cleanBatch } = this.dataLoader.getRandomTestBatch(
            this.testData.xs, this.testData.ys, 1
        );
        
        const noisyBatch = this.dataLoader.addRandomNoise(cleanBatch, this.noiseLevel);
        const reconstructed = model.predict(noisyBatch);
        
        // Create visualization
        const surface = tfvis.visor().surface({ name: 'Denoising Progress', tab: 'Training' });
        surface.drawArea.innerHTML = '';
        
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.gap = '10px';
        
        const canvases = [cleanBatch, noisyBatch, reconstructed];
        const labels = ['Original', 'Noisy', `Reconstructed (epoch ${epoch + 1})`];
        
        for (let i = 0; i < 3; i++) {
            const div = document.createElement('div');
            div.style.textAlign = 'center';
            
            const canvas = document.createElement('canvas');
            await this.dataLoader.draw28x28ToCanvas(canvases[i], canvas, 3);
            
            const label = document.createElement('div');
            label.textContent = labels[i];
            
            div.appendChild(canvas);
            div.appendChild(label);
            row.appendChild(div);
        }
        
        surface.drawArea.appendChild(row);
        
        cleanBatch.dispose();
        noisyBatch.dispose();
        reconstructed.dispose();
    }

    async calculateAccuracy(predicted, trueLabels) {
        const equals = predicted.equal(trueLabels);
        const accuracy = equals.mean();
        const result = await accuracy.data();
        equals.dispose();
        accuracy.dispose();
        return result[0];
    }

    async createConfusionMatrix(predicted, trueLabels) {
        const predArray = await predicted.array();
        const trueArray = await trueLabels.array();
        
        const matrix = Array(10).fill().map(() => Array(10).fill(0));
        
        for (let i = 0; i < predArray.length; i++) {
            const pred = predArray[i];
            const trueVal = trueArray[i];
            matrix[trueVal][pred]++;
        }
        
        return matrix;
    }

    calculatePerClassAccuracy(confusionMatrix) {
        return confusionMatrix.map((row, i) => {
            const correct = row[i];
            const total = row.reduce((sum, val) => sum + val, 0);
            return { accuracy: total > 0 ? correct / total : 0, class: i };
        });
    }

    onReset() {
        // Dispose all models
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
        
        this.dataLoader.dispose();
        this.trainData = null;
        this.testData = null;
        
        this.updateDataStatus(0, 0);
        this.updateModelInfo();
        this.clearPreview();
        this.showStatus('Reset completed');
        
        // Reset metrics display
        document.getElementById('metricPrimary').textContent = '-';
        document.getElementById('metricSecondary').textContent = '-';
    }

    toggleVisor() {
        tfvis.visor().toggle();
    }

    clearPreview() {
        document.getElementById('previewContainer').innerHTML = '';
    }

    updateDataStatus(trainCount, testCount) {
        const statusEl = document.getElementById('dataStatus');
        statusEl.innerHTML = `
            <h3>Data Status</h3>
            <p>Train samples: ${trainCount}</p>
            <p>Test samples: ${testCount}</p>
        `;
    }

    updateModelInfo() {
        const infoEl = document.getElementById('modelInfo');
        
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
        const entry = document.createElement('div');
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logs.appendChild(entry);
        logs.scrollTop = logs.scrollHeight;
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
