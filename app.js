let trainData, testData;
let preprocessedTrain, preprocessedTest;
let model;
let validationData, validationLabels, validationPredictions;
let testPredictions;

const TARGET = "Survived";
const ID = "PassengerId";

async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];

    trainData = await parseCSV(trainFile);
    testData = await parseCSV(testFile);

    document.getElementById('data-status').innerText =
        `Loaded: ${trainData.length} train / ${testData.length} test`;

    document.getElementById('preprocess-btn').disabled = false;
}

function parseCSV(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: res => resolve(res.data),
            error: err => reject(err)
        });
    });
}

function preprocessData() {

    const ageMedian = median(trainData.map(r => r.Age).filter(v => v));
    const fareMedian = median(trainData.map(r => r.Fare).filter(v => v));

    const ageStd = std(trainData.map(r => r.Age).filter(v => v));
    const fareStd = std(trainData.map(r => r.Fare).filter(v => v));

    preprocessedTrain = {x:[], y:[]};
    preprocessedTest = {x:[], ids:[]};

    trainData.forEach(r=>{
        preprocessedTrain.x.push(extract(r));
        preprocessedTrain.y.push(r[TARGET]);
    });

    testData.forEach(r=>{
        preprocessedTest.x.push(extract(r));
        preprocessedTest.ids.push(r[ID]);
    });

    function extract(r){
        const age = (r.Age ?? ageMedian);
        const fare = (r.Fare ?? fareMedian);

        const features = [
            (age-ageMedian)/ageStd,
            (fare-fareMedian)/fareStd,
            r.SibSp||0,
            r.Parch||0,
            ...oneHot(r.Pclass,[1,2,3]),
            ...oneHot(r.Sex,["male","female"]),
            ...oneHot(r.Embarked,["C","Q","S"])
        ];
        return features;
    }

    preprocessedTrain.x = tf.tensor2d(preprocessedTrain.x);
    preprocessedTrain.y = tf.tensor1d(preprocessedTrain.y);

    document.getElementById('preprocessing-output').innerText =
        `Train shape: ${preprocessedTrain.x.shape}`;

    document.getElementById('model-btn').disabled = false;
}

function createModel(){

    const dim = preprocessedTrain.x.shape[1];
    const input = tf.input({shape:[dim]});

    const gate = tf.layers.dense({
        units: dim,
        activation:'sigmoid',
        name:'gate_layer'
    }).apply(input);

    const gated = tf.layers.multiply().apply([input, gate]);

    const hidden = tf.layers.dense({units:16, activation:'relu'}).apply(gated);

    const output = tf.layers.dense({units:1, activation:'sigmoid'}).apply(hidden);

    model = tf.model({inputs:input, outputs:output});

    model.compile({
        optimizer:'adam',
        loss:'binaryCrossentropy',
        metrics:['accuracy']
    });

    document.getElementById('model-summary').innerText =
        `Params: ${model.countParams()}`;

    document.getElementById('train-btn').disabled = false;
}

async function trainModel(){

    const split = Math.floor(preprocessedTrain.x.shape[0]*0.8);

    const xTrain = preprocessedTrain.x.slice(0,split);
    const yTrain = preprocessedTrain.y.slice(0,split);

    const xVal = preprocessedTrain.x.slice(split);
    const yVal = preprocessedTrain.y.slice(split);

    validationData = xVal;
    validationLabels = yVal;

    await model.fit(xTrain,yTrain,{
        epochs:50,
        batchSize:32,
        validationData:[xVal,yVal],
        callbacks: tfvis.show.fitCallbacks(
            {name:'Training'},
            ['loss','acc','val_loss','val_acc']
        )
    });

    validationPredictions = model.predict(validationData);

    document.getElementById('threshold-slider').disabled = false;
    document.getElementById('threshold-slider')
        .addEventListener('input',updateMetrics);

    document.getElementById('predict-btn').disabled = false;

    updateMetrics();
    showFeatureImportance();
}

async function updateMetrics(){

    const threshold = parseFloat(
        document.getElementById('threshold-slider').value);

    document.getElementById('threshold-value').innerText =
        threshold.toFixed(2);

    const preds = validationPredictions.arraySync().map(v=>v[0]);
    const labels = validationLabels.arraySync();

    let tp=0,tn=0,fp=0,fn=0;

    for(let i=0;i<preds.length;i++){
        const p = preds[i]>=threshold?1:0;
        const y = labels[i];

        if(p===1 && y===1) tp++;
        else if(p===0 && y===0) tn++;
        else if(p===1 && y===0) fp++;
        else fn++;
    }

    document.getElementById('confusion-matrix').innerHTML=
        `TP:${tp} TN:${tn} FP:${fp} FN:${fn}`;

    const acc = (tp+tn)/(tp+tn+fp+fn);

    document.getElementById('performance-metrics').innerHTML=
        `Accuracy: ${(acc*100).toFixed(2)}%`;
}

async function showFeatureImportance(){

    const gate = model.getLayer('gate_layer');
    const weights = gate.getWeights()[0];
    const importance = await weights.mean(1).array();

    const data = importance.map((v,i)=>({x:`F${i}`,y:v}));

    tfvis.render.barchart(
        {name:'Feature Importance'},
        data
    );
}

async function predict(){

    const xTest = tf.tensor2d(preprocessedTest.x);
    testPredictions = model.predict(xTest);

    const preds = testPredictions.arraySync().map(v=>v[0]);

    document.getElementById('prediction-output').innerText =
        `Predictions done: ${preds.length}`;

    document.getElementById('export-btn').disabled = false;
}

function exportResults(){

    const preds = testPredictions.arraySync().map(v=>v[0]);

    let csv = "PassengerId,Survived\n";

    preprocessedTest.ids.forEach((id,i)=>{
        csv += `${id},${preds[i]>=0.5?1:0}\n`;
    });

    const link=document.createElement("a");
    link.href=URL.createObjectURL(new Blob([csv]));
    link.download="submission.csv";
    link.click();
}

function median(arr){
    arr.sort((a,b)=>a-b);
    return arr[Math.floor(arr.length/2)];
}

function std(arr){
    const mean=arr.reduce((a,b)=>a+b)/arr.length;
    return Math.sqrt(arr.map(x=>(x-mean)**2).reduce((a,b)=>a+b)/arr.length);
}

function oneHot(val,cats){
    return cats.map(c=>c===val?1:0);
}
