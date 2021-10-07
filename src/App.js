import './App.css';
import * as tf from '@tensorflow/tfjs'
import {useRef, useState} from 'react'
import SignaturePad from 'react-signature-canvas'

function App() {
  let sigRef = useRef({})
  const [prediction, setPrediction] = useState("None")
  const clearBoard = () => {
    sigRef.clear()
  }

  async function Predict () {
    const path = 'http://localhost:8080/model.json'
    // const canvas = sigRef.getTrimmedCanvas()
    const canvas = sigRef.getCanvas()
    const tensor = preprocessCanvas(canvas)
    const model = await tf.loadLayersModel(path)
    .then((actualModel)=>{
      console.log("success loading the model:\n")
      console.log("attempting to predict:\n")
      const predData = actualModel.predict(tensor)
      const confidence = predData.max()
      const pred = predData.argMax(1)
      const keys = Object.keys(pred)
      console.log("values:\n" + predData.values)
      console.log("typeof pred: " + typeof(pred))
      keys.forEach(key => {
        console.log(`pred[${key}]: ${pred[key]}`)
      })
      setPrediction(pred.dataSync()[0])
    })
    .catch((err)=>{console.log(`Error:\n ${err}`)})
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
      <SignaturePad ref={(ref)=>{sigRef=ref}} canvasProps={sigCanvasProps} 
      backgroundColor="white" minWidth = {penThickness} maxWidth={penThickness}/>
      <br/><br/>
      <button onClick={clearBoard}>Clear board</button>
      <button onClick={Predict}>Predict</button>
      <h1>Prediction is {prediction}</h1>
    </div>
  );
}
export default App;