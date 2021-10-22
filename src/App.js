import './App.css';
import * as tf from '@tensorflow/tfjs'
import {useRef, useState, useEffect} from 'react'
import SignaturePad from 'react-signature-canvas'
import {Bar} from 'react-chartjs-2'

//Digit-recognition model path, which needs to be fetched from a server
const modelPath = 'http://localhost:8080/model.json'
let model

const options = {
  scales: {
      xAxes: [{
          barPercentage: 0.4
      }]
  }
}

async function loadModel () {
  model = await tf.loadLayersModel(modelPath)
}

 function App() {
  //load the digit-recognition ML model
  useEffect(() => {
    loadModel()
    return () => {
      
    }
  }, [])

  //reference for the drawing pad
  let padRef = useRef({})
  const [prediction, setPrediction] = useState("None")
  const [hist, setHist] = useState(null)
  const clearBoard = () => {
    padRef.clear()
    setHist(null)
  }

  async function Predict () {
    const canvas = padRef.getCanvas()
    const tensor = preprocessCanvas(canvas)
    //predData.dataSync() == array of confidences/predicted probabilites for each number
    const predData = model.predict(tensor)
    const histogramData = {
      labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
      datasets: [
        {
          label: 'Confidence',
          backgroundColor: '#EC932F',
          borderColor: 'rgba(255,99,132,1)',
          borderWidth: 1,
          hoverBackgroundColor: 'rgba(255,99,132,0.4)',
          hoverBorderColor: 'rgba(255,99,132,1)',
          data: predData.dataSync()
        }
      ]
    }
    setHist(histogramData)
    //actual prediction i.e. guessed number
    const pred = predData.argMax(1)
    setPrediction(pred.dataSync()[0])
  }

  function preprocessCanvas(image) {
      // resize the input image to target size of (1, 28, 28, 1)
      let tensor = tf.browser.fromPixels(image)
      // .resizeNearestNeighbor([28, 28])
      .resizeBilinear([28,28])
      .mean(2) //mean on axis 2 (colors)
      .reshape([1, 28, 28, 1])
      .toFloat();
      let newTensor = tensor.mul(-1).add(255)
      return newTensor
        
}

  const sigCanvasProps = {
    width: 200, height: 200, className: 'sigCanvas'
  }

  const penThickness = 7

  return (
    <div className="App">
      <h1>Hand-written digit recognizer</h1> <br/>
      <p>Draw a digit from 0 to 9 on the drawing pad.</p>
      <p> Click on "Predict" to predict it. Click on "Clear board" to clear the drawing pad.</p>
      <div style={styles.signaturePad}>
        <SignaturePad ref={(ref)=>{padRef=ref}} canvasProps={sigCanvasProps} 
        backgroundColor="white" minWidth = {penThickness} maxWidth={penThickness}/>
      </div>
      
      <br/><br/>
      <button onClick={clearBoard}>Clear board</button>
      <button onClick={Predict}>Predict</button>
      <h1>Prediction is {prediction}</h1>
      <Bar data={hist} width={100} height={50} options={options}/>
    </div>
  );
}
export default App;

const margin = 570
const styles = {
  signaturePad:{
    backgroundColor: 'yellow',
    marginLeft: margin,
    marginRight: margin
  }
}