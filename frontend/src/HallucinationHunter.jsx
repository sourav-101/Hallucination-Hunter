import React, { useState, useRef } from 'react';
import {
  Upload, AlertCircle, CheckCircle, HelpCircle,
  FileText, Zap, BarChart3, Download
} from 'lucide-react';

const HallucinationHunter = () => {
  const [sourceDoc, setSourceDoc] = useState('');
  const [llmOutput, setLlmOutput] = useState('');
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const sourceRef = useRef(null);

  const loadDemoData = () => {
    setSourceDoc(`Diagnosis: Type 1 Diabetes
Medication: Insulin 20 units daily
Blood Pressure: 128/82 mmHg (slightly elevated)
Allergies: Penicillin`);
    setLlmOutput(`Patient has Type 2 Diabetes and takes 25 units of insulin daily.
Blood pressure is normal.`);
  };

  const processDocument = async () => {
    setProcessing(true);
    setResults(null);

    try {
      const res = await fetch("http://localhost:8000/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_documents: [sourceDoc],
          llm_output: llmOutput
        })
      });

      const data = await res.json();
      setResults(data);

    } catch (err) {
      alert("Backend not reachable. Is FastAPI running?");
      console.error(err);
    }

    setProcessing(false);
  };

  const exportReport = () => {
    if (!results) return;
    const blob = new Blob([JSON.stringify(results, null, 2)], {
      type: "application/json"
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "hallucination-report.json";
    a.click();
  };

  return (
    <div className="min-h-screen bg-slate-100 p-6">
      <div className="max-w-7xl mx-auto space-y-6">

        {/* Header */}
        <div className="bg-white p-6 rounded-lg shadow flex justify-between">
          <div>
            <h1 className="text-3xl font-bold flex gap-2 items-center">
              <AlertCircle className="text-blue-600" />
              Hallucination Hunter
            </h1>
            <p className="text-slate-600">AI-Powered Fact Checking</p>
          </div>
          <button
            onClick={loadDemoData}
            className="bg-blue-600 text-white px-4 py-2 rounded"
          >
            Load Demo
          </button>
        </div>

        {/* Inputs */}
        <div className="grid grid-cols-2 gap-6">
          <textarea
            value={sourceDoc}
            onChange={e => setSourceDoc(e.target.value)}
            className="h-80 p-4 border rounded"
            placeholder="Trusted source document"
          />
          <textarea
            value={llmOutput}
            onChange={e => setLlmOutput(e.target.value)}
            className="h-80 p-4 border rounded"
            placeholder="LLM output to verify"
          />
        </div>

        {/* Action */}
        <div className="text-center">
          <button
            disabled={processing}
            onClick={processDocument}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg"
          >
            {processing ? "Verifying..." : "Verify"}
          </button>
        </div>

        {/* Results */}
        {results && (
          <div className="bg-white p-6 rounded-lg shadow space-y-4">
            <div className="flex justify-between">
              <h2 className="text-xl font-semibold">
                Trust Score: {results.trustScore.toFixed(1)}%
              </h2>
              <button onClick={exportReport} className="flex gap-2">
                <Download /> Export
              </button>
            </div>

            {results.claims.map(c => (
              <div
                key={c.id}
                onClick={() => setSelectedClaim(c)}
                className={`p-4 border rounded cursor-pointer ${
                  c.status === "supported" ? "bg-green-50 border-green-300" :
                  c.status === "contradicted" ? "bg-red-50 border-red-300" :
                  "bg-yellow-50 border-yellow-300"
                }`}
              >
                <div className="font-medium">{c.text}</div>
                <div className="text-sm text-slate-600">{c.explanation}</div>
              </div>
            ))}
          </div>
        )}

        {/* Claim Detail */}
        {selectedClaim && (
          <div className="bg-white p-6 rounded-lg shadow">
            <h3 className="font-semibold mb-2">Selected Claim</h3>
            <p>{selectedClaim.text}</p>
            {selectedClaim.correction && (
              <p className="mt-2 text-green-700">
                Correction: {selectedClaim.correction}
              </p>
            )}
          </div>
        )}

      </div>
    </div>
  );
};

export default HallucinationHunter;
