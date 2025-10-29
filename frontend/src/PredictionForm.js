import { useState } from "react";

export default function PredictionForm() {
  // Track which model is expanded
  const [expandedModel, setExpandedModel] = useState("");
  const [headlineText, setHeadlineText] = useState("");

  

  const [featuresByModel, setFeaturesByModel] = useState({
    model1: [0, 0, 0, 0],
    model2: [0, 0, 0, 0],
  });

  // Track prediction result **per model**
  const [resultByModel, setResultByModel] = useState({
    model1: "",
    model2: "",
  });


  const models = [
    { model_name: "model1", modelTitle: "Iris classifier", modelDescription: "classifies the flower ..." },
    { model_name: "model2", modelTitle: "AI Topic Classifier", modelDescription: "Classifies headlines into Broad + Subcategories" },
  ];


  const handleChange = (modelName, index, value) => {
    setFeaturesByModel((prev) => ({
      ...prev,
      [modelName]: prev[modelName].map((v, i) =>
        i === index ? parseFloat(value) : v
      ),
    }));
  };

//   const predict = async (modelName) => {
//     try {
//       const response = await fetch(
//         `http://localhost:8000/predict/${modelName}`,
//         {
//           method: "POST",
//           headers: { "Content-Type": "application/json" },
//           body: JSON.stringify({ features: featuresByModel[modelName] }),
//         }
//       );
//       const data = await response.json();
//       setResultByModel((prev) => ({ ...prev, [modelName]: data.predicted_class }));
//     } catch (err) {
//       console.error("Error fetching prediction:", err);
//     }
//   };

  const predict = async (modelName) => {
    try {
        let requestBody;

        if (modelName === "model2") {
        console.log()
        requestBody = JSON.stringify({ text: headlineText });
        } else {
        requestBody = JSON.stringify({ features: featuresByModel[modelName] });
        }

    const response = await fetch(`http://localhost:8000/predict/${modelName}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestBody,
        });

        const data = await response.json();
        console.log('result by model:',data);
        setResultByModel((prev) => ({ ...prev, [modelName]: data }));
    } catch (err) {
        console.error("Error fetching prediction:", err);
    }
    };

const renderModelDemo = (item) => {
  const modelName = item.model_name;
  const isExpanded = expandedModel === modelName;

  return (
    <div
      key={modelName}
      style={styles.modelContainer}
      onClick={() => setExpandedModel(isExpanded ? "" : modelName)}
    >
      <div style={styles.modelHeader}>
        <div style={{ fontWeight: "bold" }}>{item.modelTitle}</div>
        {isExpanded ? <div>D</div> : <div>V</div>}
      </div>

      {isExpanded && (
        <div style={styles.modelDescriptionBlock} onClick={(e) => e.stopPropagation()}>
          <div style={styles.descDiv}>
            <strong>Description:</strong> {item.modelDescription}
          </div>

          {/* Model 1 input (numeric features) */}
          {modelName === "model1" && (
            <>
              {featuresByModel[modelName].map((f, idx) => (
                <input
                  key={idx}
                  type="number"
                  step="0.1"
                  value={featuresByModel[modelName][idx]}
                  onChange={(e) => handleChange(modelName, idx, e.target.value)}
                  style={{ marginRight: "5px" }}
                />
              ))}
            </>
          )}

          {/* Model 2 input (headline text) */}
          {modelName === "model2" && (
            <input
              type="text"
              placeholder="Enter AI related headline"
              value={headlineText}
              onChange={(e) => setHeadlineText(e.target.value)}
              style={{ width: "100%", marginBottom: 10 }}
            />
          )}

          {/* Buttons */}
          <button onClick={() => predict(modelName)}>Predict</button>
          <button
            onClick={() => {
              if (modelName === "model2") setHeadlineText("");
              else setFeaturesByModel((prev) => ({ ...prev, [modelName]: [0, 0, 0, 0] }));
              setResultByModel((prev) => ({ ...prev, [modelName]: null }));
            }}
          >
            Reset
          </button>

          {/* Prediction display */}
          {resultByModel[modelName] && (
            <div style={{ marginTop: 10 }}>
              {modelName === "model1" ? (
                <strong>Prediction: {resultByModel[modelName]?.predicted_class}</strong>
              ) : (
                <ul style={{ backgroundColor: "#fdd", padding: "10px", borderRadius: "0.5rem" }}>
                  {resultByModel[modelName]?.predictions?.map((p, i) => (
                    <li key={i} style={{ color: "#000" }}>
                      {p} — {(resultByModel[modelName].probabilities[i] * 100).toFixed(2)}%
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};


  return (
    <div style={{ padding: "20px" }}>
      <h2>ML models from project works</h2>
      {models.map((item) => renderModelDemo(item))}
    </div>
  );
}

const styles = {
  modelContainer: {
    border: "1px solid #ccc",
    marginBottom: "10px",
    cursor: "pointer",
    borderRadius:"1rem",
    overflow:'hidden'
  },
  modelDescriptionBlock: {

    padding:15
  },
  modelHeader: {
    display:'flex',
    flexDirection:'row',
    justifyContent: 'space-between',
    padding:15,
    backgroundColor:'skyblue',
  }
};
