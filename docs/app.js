/********************************************************************
 MAIN APPLICATION LOGIC

 This file wires together:
 - UI controls
 - TensorFlow.js models
 - Training
 - Evaluation
 - Visualization
 - Model saving/loading

 Supports TWO major modes:

 1) Classification (digit recognition)
 2) Denoising Autoencoder (MaxPooling / AvgPooling)

********************************************************************/

let trainData = null;
let testData = null;

let classificationModel = null;
let modelMax = null;
let modelAvg = null;

let currentMode = "classification";
let noiseLevel = 0.3;

/********************************************************************
 UI REFERENCES
********************************************************************/

const logBox = document.getElementById("logBox");
const dataStatus = document.getElementById("dataStatus");
const metricsBox = document.getElementById("metrics");
const previewArea = document.getElementById("previewArea");
const modelInfo = document.getElementById("modelInfo");

/********************************************************************
 LOGGING HELPER
********************************************************************/
function log(msg){

  logBox.innerText += msg + "\n";
  logBox.scrollTop = logBox.scrollHeight;

}

/********************************************************************
 NOISE SLIDER
********************************************************************/
document.getElementById("noiseSlider").oninput = e=>{

  noiseLevel = Number(e.target.value);
  document.getElementById("noiseValue").innerText = noiseLevel.toFixed(2);

};

/********************************************************************
 MODE SELECTION
********************************************************************/
document.querySelectorAll("input[name=mode]").forEach(r=>{

  r.onchange = e=>{
    currentMode = e.target.value;
    log("Mode changed to " + currentMode);
  };

});

/********************************************************************
 LOAD DATA
********************************************************************/
document.getElementById("loadDataBtn").onclick = async ()=>{

  try{

    const trainFile = document.getElementById("trainFile").files[0];
    const testFile = document.getElementById("testFile").files[0];

    log("Loading training data...");
    trainData = await loadTrainFromFiles(trainFile);

    log("Loading test data...");
    testData = await loadTestFromFiles(testFile);

    dataStatus.innerText =
      "Train: " + trainData.xs.shape[0] +
      " images | Test: " + testData.xs.shape[0] + " images";

  }catch(e){
    alert("Error loading data: " + e);
  }

};

/********************************************************************
 CLASSIFICATION MODEL
********************************************************************/
function buildClassifier(){

  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    filters:32,
    kernelSize:3,
    activation:'relu',
    padding:'same',
    inputShape:[28,28,1]
  }));

  model.add(tf.layers.conv2d({
    filters:64,
    kernelSize:3,
    activation:'relu',
    padding:'same'
  }));

  model.add(tf.layers.maxPooling2d({poolSize:2}));

  model.add(tf.layers.dropout({rate:0.25}));

  model.add(tf.layers.flatten());

  model.add(tf.layers.dense({units:128,activation:'relu'}));

  model.add(tf.layers.dropout({rate:0.5}));

  model.add(tf.layers.dense({units:10,activation:'softmax'}));

  model.compile({
    optimizer:tf.train.adam(0.001),
    loss:'categoricalCrossentropy',
    metrics:['accuracy']
  });

  return model;

}

/********************************************************************
 AUTOENCODER BUILDER

 poolType:
   "max"
   "avg"
********************************************************************/
function buildAutoencoder(poolType){

  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    filters:32,
    kernelSize:3,
    activation:'relu',
    padding:'same',
    inputShape:[28,28,1]
  }));

  if(poolType==="max")
    model.add(tf.layers.maxPooling2d({poolSize:2}));
  else
    model.add(tf.layers.averagePooling2d({poolSize:2}));

  model.add(tf.layers.conv2d({
    filters:64,
    kernelSize:3,
    activation:'relu',
    padding:'same'
  }));

  if(poolType==="max")
    model.add(tf.layers.maxPooling2d({poolSize:2}));
  else
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
    optimizer:tf.train.adam(0.001),
    loss:'meanSquaredError'
  });

  return model;

}

/********************************************************************
 TRAIN
********************************************************************/
document.getElementById("trainBtn").onclick = async ()=>{

  if(!trainData) return alert("Load data first.");

  if(currentMode==="classification"){

    classificationModel = buildClassifier();

    const {trainXs,trainYs,valXs,valYs} =
      splitTrainVal(trainData.xs,trainData.ys);

    await classificationModel.fit(trainXs,trainYs,{
      epochs:5,
      batchSize:128,
      validationData:[valXs,valYs],
      callbacks:tfvis.show.fitCallbacks(
        {name:"Training"},
        ["loss","val_loss","acc","val_acc"]
      )
    });

    log("Classifier training complete.");

  }

  if(currentMode==="denoising-max"){

    modelMax = buildAutoencoder("max");

    const noisy = addRandomNoise(trainData.xs,noiseLevel);

    await modelMax.fit(noisy,trainData.xs,{
      epochs:10,
      batchSize:64,
      validationSplit:0.1,
      callbacks:tfvis.show.fitCallbacks(
        {name:"Denoiser MaxPool"},
        ["loss","val_loss"]
      )
    });

  }

  if(currentMode==="denoising-avg"){

    modelAvg = buildAutoencoder("avg");

    const noisy = addRandomNoise(trainData.xs,noiseLevel);

    await modelAvg.fit(noisy,trainData.xs,{
      epochs:10,
      batchSize:64,
      validationSplit:0.1,
      callbacks:tfvis.show.fitCallbacks(
        {name:"Denoiser AvgPool"},
        ["loss","val_loss"]
      )
    });

  }

};

/********************************************************************
 TEST 5 RANDOM
********************************************************************/
document.getElementById("testFiveBtn").onclick = async ()=>{

  if(!testData) return alert("Load test data.");

  previewArea.innerHTML = "";

  const {xsBatch} = getRandomTestBatch(testData.xs,testData.ys,5);

  const noisyBatch = addRandomNoise(xsBatch,noiseLevel);

  for(let i=0;i<5;i++){

    const row = document.createElement("div");
    row.className="preview-row";

    const orig = xsBatch.slice([i,0,0,0],[1,28,28,1]).squeeze();
    const noisy = noisyBatch.slice([i,0,0,0],[1,28,28,1]).squeeze();

    const block1 = createCanvasBlock("Original",orig);
    const block2 = createCanvasBlock("Noisy",noisy);

    row.appendChild(block1);
    row.appendChild(block2);

    if(modelMax){

      const pred = modelMax.predict(noisy.expandDims(0)).squeeze();
      const mse = calculateMSE(orig,pred);

      row.appendChild(createCanvasBlock("MaxPool "+mse.toFixed(4),pred));

    }

    if(modelAvg){

      const pred = modelAvg.predict(noisy.expandDims(0)).squeeze();
      const mse = calculateMSE(orig,pred);

      row.appendChild(createCanvasBlock("AvgPool "+mse.toFixed(4),pred));

    }

    previewArea.appendChild(row);

  }

};

/********************************************************************
 CREATE CANVAS BLOCK
********************************************************************/
function createCanvasBlock(label,tensor){

  const div = document.createElement("div");
  div.className="preview-block";

  const canvas = document.createElement("canvas");

  draw28x28ToCanvas(tensor,canvas,4);

  const txt = document.createElement("div");
  txt.innerText = label;

  div.appendChild(canvas);
  div.appendChild(txt);

  return div;

}

/********************************************************************
 COMPARE POOLING
********************************************************************/
document.getElementById("comparePoolingBtn").onclick = async ()=>{

  log("Training MaxPooling autoencoder...");
  modelMax = buildAutoencoder("max");

  await modelMax.fit(
    addRandomNoise(trainData.xs,noiseLevel),
    trainData.xs,
    {epochs:10,batchSize:64}
  );

  log("Training AveragePooling autoencoder...");
  modelAvg = buildAutoencoder("avg");

  await modelAvg.fit(
    addRandomNoise(trainData.xs,noiseLevel),
    trainData.xs,
    {epochs:10,batchSize:64}
  );

  log("Comparison training complete.");

  document.getElementById("testFiveBtn").click();

};

/********************************************************************
 EVALUATE
********************************************************************/
document.getElementById("evaluateBtn").onclick = async ()=>{

  if(currentMode==="classification"){

    const preds = classificationModel.predict(testData.xs);

    const predLabels = preds.argMax(-1);
    const trueLabels = testData.ys.argMax(-1);

    const acc = predLabels.equal(trueLabels).mean().dataSync()[0];

    metricsBox.innerText = "Accuracy: " + acc.toFixed(4);

  }

  else{

    const noisy = addRandomNoise(testData.xs,noiseLevel);

    let model = currentMode==="denoising-max" ? modelMax : modelAvg;

    const recon = model.predict(noisy);

    const mse = calculateMSE(testData.xs,recon);

    metricsBox.innerText = "MSE: " + mse.toFixed(6);

  }

};

/********************************************************************
 SAVE MODEL
********************************************************************/
document.getElementById("saveBtn").onclick = async ()=>{

  let model=null;
  let name="";

  if(currentMode==="classification"){
    model=classificationModel;
    name="mnist-classifier";
  }

  if(currentMode==="denoising-max"){
    model=modelMax;
    name="mnist-denoiser-max";
  }

  if(currentMode==="denoising-avg"){
    model=modelAvg;
    name="mnist-denoiser-avg";
  }

  if(!model) return alert("No model to save.");

  await model.save("downloads://"+name);

};

/********************************************************************
 LOAD MODEL
********************************************************************/
document.getElementById("loadModelBtn").onclick = async ()=>{

  const json = document.getElementById("modelJson").files[0];
  const weights = document.getElementById("modelWeights").files[0];

  const model = await tf.loadLayersModel(
    tf.io.browserFiles([json,weights])
  );

  const layerNames = model.layers.map(l=>l.getClassName());

  if(layerNames.includes("Flatten")){
    classificationModel=model;
    currentMode="classification";
  }

  if(layerNames.includes("MaxPooling2D")){
    modelMax=model;
    currentMode="denoising-max";
  }

  if(layerNames.includes("AveragePooling2D")){
    modelAvg=model;
    currentMode="denoising-avg";
  }

  log("Model loaded. Mode: "+currentMode);

};

/********************************************************************
 RESET
********************************************************************/
document.getElementById("resetBtn").onclick = ()=>{

  tf.disposeVariables();

  classificationModel=null;
  modelMax=null;
  modelAvg=null;

  previewArea.innerHTML="";
  metricsBox.innerHTML="";
  logBox.innerHTML="";

  log("Reset complete.");

};

/********************************************************************
 TOGGLE VISOR
********************************************************************/
document.getElementById("toggleVisorBtn").onclick = ()=>{

  tfvis.visor().toggle();

};
