import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowUpRight,
  BrainCircuit,
  Check,
  CloudUpload,
  Image as ImageIcon,
  LoaderCircle,
  RotateCcw,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import { classifyImage } from "./api";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/avif"];
const ACCEPTED_EXTENSIONS = ["jpg", "jpeg", "png", "avif"];

function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function selectFile(nextFile) {
    if (!nextFile) return;
    const extension = nextFile.name.split(".").pop()?.toLowerCase();
    if (!ACCEPTED_TYPES.includes(nextFile.type) && !ACCEPTED_EXTENSIONS.includes(extension)) {
      setError("Please choose a JPG, JPEG, PNG, or AVIF image.");
      return;
    }
    if (nextFile.size > 10 * 1024 * 1024) {
      setError("Please choose an image smaller than 10 MB.");
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setResult(null);
    setError("");
  }

  function clearImage() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl("");
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  async function handleClassify() {
    if (!file || isLoading) return;
    setIsLoading(true);
    setError("");
    try {
      setResult(await classifyImage(file));
    } catch (classificationError) {
      setError(classificationError.message);
    } finally {
      setIsLoading(false);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  }

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="app-container">
        <header className="topbar">
          <div className="brand-lockup">
            <div className="brand-mark"><BrainCircuit size={21} strokeWidth={2.4} /></div>
            <div>
              <p className="brand-name">CatVision V1.0</p>
              <p className="brand-subtitle">Cat Classifier using Linear Regression</p>
            </div>
          </div>
          <div className="header-status"><span className="status-dot" /> Model online</div>
        </header>

        <section className="hero-copy">
          <div className="eyebrow"><Sparkles size={14} /> Computer vision, made clear</div>
          <h1>See what your image<br /><span>is telling you.</span></h1>
          <p>Upload an image and let the deep neural network produce a prediction with confidence scores for every class.</p>
        </section>

        <section className="workspace-grid">
          <div className="panel upload-panel">
            <div className="panel-heading">
              <div>
                <p className="section-kicker">01 / Input</p>
                <h2>Upload an image</h2>
              </div>
              <span className="step-icon"><CloudUpload size={19} /></span>
            </div>
            <div
              className={`drop-zone ${isDragging ? "is-dragging" : ""} ${file ? "has-file" : ""}`}
              onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => event.key === "Enter" && inputRef.current?.click()}
            >
              <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.avif,image/jpeg,image/png,image/avif" hidden onChange={(event) => selectFile(event.target.files?.[0])} />
              <div className="upload-orb"><CloudUpload size={28} /></div>
              <h3>{file ? file.name : "Drop your image here"}</h3>
              <p>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB ready to analyze` : "or click to browse from your computer"}</p>
              {!file && <span className="browse-button">Browse files <ArrowUpRight size={15} /></span>}
            </div>
            <div className="format-note"><ShieldCheck size={15} /> JPG, JPEG, PNG, or AVIF · max 10 MB</div>
          </div>

          <div className="panel preview-panel">
            <div className="panel-heading">
              <div>
                <p className="section-kicker">02 / Preview</p>
                <h2>Your image</h2>
              </div>
              {file && <button className="text-button" onClick={clearImage}><RotateCcw size={14} /> Choose another</button>}
            </div>
            <div className={`preview-frame ${!previewUrl ? "empty" : ""}`}>
              {previewUrl ? <img src={previewUrl} alt="Selected image preview" /> : <div className="empty-preview"><ImageIcon size={30} /><p>Your selected image<br />will appear here</p></div>}
            </div>
            <button className="classify-button" disabled={!file || isLoading} onClick={handleClassify}>
              {isLoading ? <><LoaderCircle className="spin" size={18} /> Analyzing image...</> : <><ScanSearch size={18} /> Classify image</>}
            </button>
          </div>
        </section>

        {error && <div className="message error-message"><AlertCircle size={19} /><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error"><X size={17} /></button></div>}

        <section className={`result-section ${result ? "is-visible" : ""}`}>
          <div className="result-heading"><div><p className="section-kicker">03 / Result</p><h2>Classification insight</h2></div>{result && <span className="verified-badge"><Check size={14} /> Analysis complete</span>}</div>
          {result ? <div className="result-grid">
            <div className="result-image"><img src={previewUrl} alt="Classified upload" /><div className="image-caption"><ImageIcon size={15} /> {file?.name}</div></div>
            <div className={`result-card ${result.isCat ? "cat-result" : "not-cat-result"}`}>
              <div className="result-icon">{result.isCat ? "🐈" : "◌"}</div>
              <p className="result-label">Prediction</p>
              <h3>{result.label}</h3>
              <p className="result-status">{result.isCat ? "A cat-like subject was detected in this image." : "No cat-like subject was detected in this image."}</p>
              <div className="confidence-row"><span>Confidence</span><strong>{result.confidence.toFixed(1)}%</strong></div>
              <div className="confidence-track"><div style={{ width: `${result.confidence}%` }} /></div>
              <button className="reset-button" onClick={clearImage}><RotateCcw size={15} /> Try another image</button>
            </div>
          </div> : <div className="result-empty"><div className="result-empty-icon"><Sparkles size={21} /></div><div><h3>Your result will appear here</h3><p>Upload an image and run the model to see its prediction.</p></div></div>}
        </section>

        <footer><span>CatVision V1.0</span><span>Powered by Md Akram Khan</span></footer>
      </div>
    </main>
  );
}

export default App;
