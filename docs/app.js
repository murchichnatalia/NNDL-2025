/********************************************************************
 MAIN APP
********************************************************************/

let trainData=null;
let testData=null;

let classificationModel=null;
let modelMax=null;
let modelAvg=null;

let currentMode="classification";
let noiseLevel=0.3;

const logBox=document.getElementById("logBox");
const previewArea=document.getElementById("previewArea");
const metricsBox=document.getElementById("metrics");
const dataStatus=document.getElementById("dataStatus");

function log(t){
  logBox.innerText+=t+"\n";
  logBox.scrollTop=logBox.scrollHeight;
}

/********************************************************************
 UI
********************************************************************/

document.getElementById("noiseSlider").oninput=e=>{
  noiseLevel=Number(e.target.value);
  document.getElementById("noiseValue").innerText=noiseLevel.toFixed(2);
};

document.querySelectorAll("input[name=mode]").forEach(r=>{
  r.onchange=e=>currentMode=e.target.value;
});

/********************************************************************
 LOAD DATA
********************************************************************/

document.getElementById("loadDataBtn").onclick=async()=>{

  try{

    const trainFile=document.getElementById("trainFile").files[0];
    const testFile=document.getElementById("testFile").files[0];

    log("Loading train data...");

    trainData=await DataLoader.loadTrainFromFiles(trainFile);

    log("Loading test data...");

    testData=await DataLoader.loadTestFromFiles(testFile);

    dataStatus.innerText=
      "Train: "+trainData.xs.shape[0]+" images | "+
      "Test: "+testData.xs.shape[0]+" images";

  }catch(e){
    alert("Error loading data: "+e);
  }

};

/********************************************************************
 MODELS
********************************************************************/

function buildClassifier(){

  const m=tf.sequential();

  m.add(tf.layers.conv2d({
    filters:32,kernelSize:3,activation:'relu',padding:'same',inputShape:[28,28,1]
  }));

  m.add(tf.layers.conv2d({
    filters:64,kernelSize:3,activation:'relu',padding:'same'
  }));

  m.add(tf.layers.maxPooling2d({poolSize:2}));

  m.add(tf.layers.flatten());

  m.add(tf.layers.dense({units:128,activation:'relu'}));

  m.add(tf.layers.dense({units:10,activation:'softmax'}));

  m.compile({
    optimizer:tf.train.adam(0.001),
    loss:'categoricalCrossentropy',
    metrics:['accuracy']
  });

  return m;
}

function buildAutoencoder(type){

  const m=tf.sequential();

  m.add(tf.layers.conv2d({
    filters:32,kernelSize:3,activation:'relu',padding:'same',inputShape:[28,28,1]
  }));

  if(type==="max")
    m.add(tf.layers.maxPooling2d({poolSize:2}));
  else
    m.add(tf.layers.averagePooling2d({poolSize:2}));

  m.add(tf.layers.conv2d({
    filters:64,kernelSize:3,activation:'relu',padding:'same'
  }));

  if(type==="max")
    m.add(tf.layers.maxPooling2d({poolSize:2}));
  else
    m.add(tf.layers.averagePooling2d({poolSize:2}));

  m.add(tf.layers.conv2d({filters:64,kernelSize:3,activation:'relu',padding:'same'}));
  m.add(tf.layers.upSampling2d({size:2}));

  m.add(tf.layers.conv2d({filters:32,kernelSize:3,activation:'relu',padding:'same'}));
  m.add(tf.layers.upSampling2d({size:2}));

  m.add(tf.layers.conv2d({filters:1,kernelSize:3,activation:'sigmoid',padding:'same'}));

  m.compile({
    optimizer:tf.train.adam(0.001),
    loss:'meanSquaredError'
  });

  return m;
}

/********************************************************************
 TRAIN
********************************************************************/

document.getElementById("trainBtn").onclick=async()=>{

  if(!trainData) return alert("Load data first");

  if(currentMode==="classification"){

    classificationModel=buildClassifier();

    const {trainXs,trainYs,valXs,valYs}=
      DataLoader.splitTrainVal(trainData.xs,trainData.ys);

    await classificationModel.fit(trainXs,trainYs,{
      epochs:5,
      batchSize:128,
      validationData:[valXs,valYs]
    });

    log("Classifier trained");
  }

  if(currentMode==="denoising-max"){

    modelMax=buildAutoencoder("max");

    const noisy=DataLoader.addRandomNoise(trainData.xs,noiseLevel);

    await modelMax.fit(noisy,trainData.xs,{
      epochs:10,
      batchSize:64
    });

    log("MaxPool autoencoder trained");
  }

  if(currentMode==="denoising-avg"){

    modelAvg=buildAutoencoder("avg");

    const noisy=DataLoader.addRandomNoise(trainData.xs,noiseLevel);

    await modelAvg.fit(noisy,trainData.xs,{
      epochs:10,
      batchSize:64
    });

    log("AvgPool autoencoder trained");
  }

};

/********************************************************************
 TEST RANDOM
********************************************************************/

document.getElementById("testFiveBtn").onclick=()=>{

  if(!testData) return;

  previewArea.innerHTML="";

  const {xsBatch}=DataLoader.getRandomTestBatch(testData.xs,testData.ys,5);

  const noisy=DataLoader.addRandomNoise(xsBatch,noiseLevel);

  for(let i=0;i<5;i++){

    const row=document.createElement("div");
    row.className="preview-row";

    const orig=xsBatch.slice([i,0,0,0],[1,28,28,1]).squeeze();
    const n=noisy.slice([i,0,0,0],[1,28,28,1]).squeeze();

    row.appendChild(drawBlock("Original",orig));
    row.appendChild(drawBlock("Noisy",n));

    if(modelMax){

      const pred=modelMax.predict(n.expandDims(0)).squeeze();
      const mse=DataLoader.calculateMSE(orig,pred);

      row.appendChild(drawBlock("Max "+mse.toFixed(4),pred));
    }

    if(modelAvg){

      const pred=modelAvg.predict(n.expandDims(0)).squeeze();
      const mse=DataLoader.calculateMSE(orig,pred);

      row.appendChild(drawBlock("Avg "+mse.toFixed(4),pred));
    }

    previewArea.appendChild(row);
  }

};

function drawBlock(label,t){

  const div=document.createElement("div");
  div.className="preview-block";

  const canvas=document.createElement("canvas");

  DataLoader.draw28x28ToCanvas(t,canvas,4);

  const txt=document.createElement("div");
  txt.innerText=label;

  div.appendChild(canvas);
  div.appendChild(txt);

  return div;
}

/********************************************************************
 SAVE MODEL
********************************************************************/

document.getElementById("saveBtn").onclick=async()=>{

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

  if(!model) return alert("No model");

  await model.save("downloads://"+name);
};

/********************************************************************
 RESET
********************************************************************/

document.getElementById("resetBtn").onclick=()=>{

  tf.disposeVariables();

  classificationModel=null;
  modelMax=null;
  modelAvg=null;

  previewArea.innerHTML="";
  metricsBox.innerHTML="";
  logBox.innerHTML="";

  log("Reset done");
};

/********************************************************************
 VISOR
********************************************************************/

document.getElementById("toggleVisorBtn").onclick=()=>{
  tfvis.visor().toggle();
};
