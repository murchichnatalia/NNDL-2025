class MNISTApp {
    constructor() {
        this.dataLoader = new MNISTDataLoader();

        this.modelMax = null;
        this.modelAvg = null;

        this.isTraining = false;
        this.trainData = null;
        this.testData = null;
        
        this.initializeUI();
    }

    initializeUI() {
        document.getElementById('loadDataBtn').addEventListener('click', () => this.onLoadData());
        document.getElementById('trainBtn').addEventListener('click', () => this.onTrain());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.onEvaluate());
        document.getElementById('testFiveBtn').addEventListener('click', () => this.onTestFive());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.onSaveDownload());
        document.getElementById('loadModelBtn').addEventListener('click', () => this.onLoadFromFiles());
        document.getElementById('resetBtn').addEventListener('click', () => this.onReset());
        document.getElementById('toggleVisorBtn').addEventListener('click', () => this.toggleVisor());
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

            this.updateDataStatus(trainData.count, testData.count);
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

            const { trainXs, valXs } =
                this.dataLoader.splitTrainVal(this.trainData.xs, this.trainData.ys, 0.1);

            const noisyTrain = this.dataLoader.addRandomNoise(trainXs,0.3);
            const noisyVal = this.dataLoader.addRandomNoise(valXs,0.3);

            this.modelMax = this.createAutoencoderMax();
            this.modelAvg = this.createAutoencoderAvg();

            this.showStatus("Training MaxPool autoencoder...");

            await this.modelMax.fit(noisyTrain, trainXs, {
                epochs:10,
                batchSize:128,
                validationData:[noisyVal,valXs],
                shuffle:true
            });

            this.showStatus("Training AvgPool autoencoder...");

            await this.modelAvg.fit(noisyTrain, trainXs, {
                epochs:10,
                batchSize:128,
                validationData:[noisyVal,valXs],
                shuffle:true
            });

            noisyTrain.dispose();
            noisyVal.dispose();
            trainXs.dispose();
            valXs.dispose();

            this.showStatus("Training finished");

        } catch(error) {

            this.showError(`Training failed: ${error.message}`);

        } finally {

            this.isTraining=false;

        }
    }

    async onEvaluate() {
        this.showStatus("Evaluation not used for autoencoder");
    }

    async onTestFive() {

        if (!this.modelMax || !this.modelAvg) {
            this.showError("Train model first");
            return;
        }

        const { batchXs } =
            this.dataLoader.getRandomTestBatch(this.testData.xs,this.testData.ys,5);

        const noisy = this.dataLoader.addRandomNoise(batchXs,0.3);

        const maxPred = this.modelMax.predict(noisy);
        const avgPred = this.modelAvg.predict(noisy);

        const orig = batchXs.arraySync();
        const noisyArr = noisy.arraySync();
        const maxArr = maxPred.arraySync();
        const avgArr = avgPred.arraySync();

        const container = document.getElementById("previewContainer");
        container.innerHTML="";

        for(let i=0;i<5;i++){

            const row=document.createElement("div");
            row.className="preview-row";

            const imgs=[orig[i],noisyArr[i],maxArr[i],avgArr[i]];
            const labels=["Original","Noisy","MaxPool","AvgPool"];

            for(let j=0;j<4;j++){

                const item=document.createElement("div");
                item.className="preview-item";

                const canvas=document.createElement("canvas");
                const label=document.createElement("div");
                label.textContent=labels[j];

                this.dataLoader.draw28x28ToCanvas(tf.tensor(imgs[j]),canvas,4);

                item.appendChild(canvas);
                item.appendChild(label);

                row.appendChild(item);
            }

            container.appendChild(row);
        }

        batchXs.dispose();
        noisy.dispose();
        maxPred.dispose();
        avgPred.dispose();
    }

    async onSaveDownload() {

        if (!this.modelMax || !this.modelAvg) {
            this.showError("No trained models");
            return;
        }

        await this.modelMax.save('downloads://mnist-denoiser-max');
        await this.modelAvg.save('downloads://mnist-denoiser-avg');

        this.showStatus("Models saved");
    }

    async onLoadFromFiles() {

        const jsonFile=document.getElementById('modelJsonFile').files[0];
        const weightsFile=document.getElementById('modelWeightsFile').files[0];

        if(!jsonFile||!weightsFile){
            this.showError("Select model files");
            return;
        }

        this.modelMax=await tf.loadLayersModel(
            tf.io.browserFiles([jsonFile,weightsFile])
        );

        this.showStatus("Model loaded");
    }

    onReset() {

        if(this.modelMax) this.modelMax.dispose();
        if(this.modelAvg) this.modelAvg.dispose();

        this.modelMax=null;
        this.modelAvg=null;

        this.dataLoader.dispose();
        this.trainData=null;
        this.testData=null;

        this.updateDataStatus(0,0);
        this.clearPreview();

        this.showStatus("Reset completed");
    }

    toggleVisor(){
        tfvis.visor().toggle();
    }

    createAutoencoderMax(){

        const model=tf.sequential();

        model.add(tf.layers.conv2d({
            filters:32,
            kernelSize:3,
            activation:'relu',
            padding:'same',
            inputShape:[28,28,1]
        }));

        model.add(tf.layers.maxPooling2d({poolSize:2}));

        model.add(tf.layers.conv2d({
            filters:64,
            kernelSize:3,
            activation:'relu',
            padding:'same'
        }));

        model.add(tf.layers.maxPooling2d({poolSize:2}));

        model.add(tf.layers.conv2d({
            filters:64,
            kernelSize:3,
            activation:'relu',
            padding:'same'
        }));

        model.add(tf.layers.upSampling2d({size:2}));

        model.add(tf.layers.conv2d({
            filters:32,
            kernelSize:3,
            activation:'relu',
            padding:'same'
        }));

        model.add(tf.layers.upSampling2d({size:2}));

        model.add(tf.layers.conv2d({
            filters:1,
            kernelSize:3,
            activation:'sigmoid',
            padding:'same'
        }));

        model.compile({
            optimizer:'adam',
            loss:'meanSquaredError'
        });

        return model;
    }

    createAutoencoderAvg(){

        const model=tf.sequential();

        model.add(tf.layers.conv2d({
            filters:32,
            kernelSize:3,
            activation:'relu',
            padding:'same',
            inputShape:[28,28,1]
        }));

        model.add(tf.layers.averagePooling2d({poolSize:2}));

        model.add(tf.layers.conv2d({
            filters:64,
            kernelSize:3,
            activation:'relu',
            padding:'same'
        }));

        model.add(tf.layers.averagePooling2d({poolSize:2}));

        model.add(tf.layers.conv2d({
            filters:64,
            kernelSize:3,
            activation:'relu',
            padding:'same'
        }));

        model.add(tf.layers.upSampling2d({size:2}));

        model.add(tf.layers.conv2d({
            filters:32,
            kernelSize:3,
            activation:'relu',
            padding:'same'
        }));

        model.add(tf.layers.upSampling2d({size:2}));

        model.add(tf.layers.conv2d({
            filters:1,
            kernelSize:3,
            activation:'sigmoid',
            padding:'same'
        }));

        model.compile({
            optimizer:'adam',
            loss:'meanSquaredError'
        });

        return model;
    }

    clearPreview(){
        document.getElementById('previewContainer').innerHTML='';
    }

    updateDataStatus(trainCount,testCount){

        const statusEl=document.getElementById('dataStatus');

        statusEl.innerHTML=`
        <h3>Data Status</h3>
        <p>Train samples: ${trainCount}</p>
        <p>Test samples: ${testCount}</p>
        `;
    }

    showStatus(message){

        const logs=document.getElementById('trainingLogs');

        const entry=document.createElement('div');
        entry.textContent=`[${new Date().toLocaleTimeString()}] ${message}`;

        logs.appendChild(entry);
        logs.scrollTop=logs.scrollHeight;
    }

    showError(message){
        this.showStatus(`ERROR: ${message}`);
        console.error(message);
    }
}

document.addEventListener('DOMContentLoaded',()=>{
    new MNISTApp();
});
