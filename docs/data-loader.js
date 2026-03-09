/********************************************************************
 DATA LOADER
********************************************************************/

const IMAGE_SIZE = 28*28;
const NUM_CLASSES = 10;

async function parseCSV(file){

  const text = await file.text();
  const lines = text.split(/\r?\n/);

  const images=[];
  const labels=[];

  for(const line of lines){

    if(!line.trim()) continue;

    const parts=line.split(",");

    if(parts.length!==785) continue;

    labels.push(Number(parts[0]));

    const pixels = parts.slice(1).map(v=>Number(v)/255);

    images.push(pixels);

  }

  const xs=tf.tensor2d(images).reshape([images.length,28,28,1]);

  const ys=tf.oneHot(
    tf.tensor1d(labels,"int32"),
    NUM_CLASSES
  );

  return {xs,ys};

}

async function loadTrainFromFiles(file){
  return await parseCSV(file);
}

async function loadTestFromFiles(file){
  return await parseCSV(file);
}

function splitTrainVal(xs,ys,valRatio=0.1){

  const num=xs.shape[0];
  const valSize=Math.floor(num*valRatio);

  const trainXs=xs.slice([0,0,0,0],[num-valSize,28,28,1]);
  const valXs=xs.slice([num-valSize,0,0,0],[valSize,28,28,1]);

  const trainYs=ys.slice([0,0],[num-valSize,10]);
  const valYs=ys.slice([num-valSize,0],[valSize,10]);

  return {trainXs,trainYs,valXs,valYs};

}

function getRandomTestBatch(xs,ys,k=5){

  const total=xs.shape[0];

  const ids=[];

  for(let i=0;i<k;i++)
    ids.push(Math.floor(Math.random()*total));

  const xsBatch=tf.stack(
    ids.map(i=>xs.slice([i,0,0,0],[1,28,28,1]).squeeze())
  );

  const ysBatch=tf.stack(
    ids.map(i=>ys.slice([i,0],[1,10]).squeeze())
  );

  return {xsBatch,ysBatch};

}

function addRandomNoise(tensor,noiseFactor=0.3){

  return tf.tidy(()=>{

    const noise=tf.randomNormal(tensor.shape);
    const noisy=tensor.add(noise.mul(noiseFactor));

    return noisy.clipByValue(0,1);

  });

}

function draw28x28ToCanvas(tensor,canvas,scale=4){

  const data=tensor.dataSync();

  canvas.width=28*scale;
  canvas.height=28*scale;

  const ctx=canvas.getContext("2d");

  const img=ctx.createImageData(28,28);

  for(let i=0;i<data.length;i++){

    const v=data[i]*255;

    img.data[i*4]=v;
    img.data[i*4+1]=v;
    img.data[i*4+2]=v;
    img.data[i*4+3]=255;

  }

  const tmp=document.createElement("canvas");
  tmp.width=28;
  tmp.height=28;

  tmp.getContext("2d").putImageData(img,0,0);

  ctx.imageSmoothingEnabled=false;
  ctx.drawImage(tmp,0,0,28*scale,28*scale);

}

function calculateMSE(original,reconstructed){

  return tf.tidy(()=>{

    const mse=tf.losses.meanSquaredError(original,reconstructed);

    return mse.mean().dataSync()[0];

  });

}

/********************************************************************
 GLOBAL EXPORT (FIX)
********************************************************************/

window.DataLoader={
  loadTrainFromFiles,
  loadTestFromFiles,
  splitTrainVal,
  getRandomTestBatch,
  addRandomNoise,
  draw28x28ToCanvas,
  calculateMSE
};
