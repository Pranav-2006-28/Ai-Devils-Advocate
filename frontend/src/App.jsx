import { useState, useRef, useCallback } from 'react'
import './index.css'

// Requests go through the Vite proxy (/api/* → localhost:8000)
const API_BASE = '/api'

const LOADING_STEPS = [
  'Extracting text from PDF…',
  'Loading AI model weights…',
  'Scanning for risk clauses…',
  'Querying ChromaDB for historical precedents…',
  'Ollama generating Lawyer, Optimist & Pessimist takes…',
  'Finalizing risk report…',
]

// ─── Filter options ───────────────────────────────────────────
const FILTERS = [
  { key: 'all', label: 'All Clauses' },
  { key: 'found', label: '🚨 Risks Found' },
  { key: 'high', label: '🔴 High Severity' },
  { key: 'medium', label: '🟡 Medium Severity' },
  { key: 'low', label: '🟢 Low Severity' },
]

// ─── Helper ───────────────────────────────────────────────────
function getRiskLevel(score) {
  if (score >= 60) return 'high'
  if (score >= 30) return 'medium'
  return 'low'
}

function getRiskLabel(score) {
  if (score >= 60) return 'High Risk'
  if (score >= 30) return 'Moderate Risk'
  return 'Low Risk'
}

// ─── Components ───────────────────────────────────────────────
function Header() {
  return (
    <header className="header">
      <div className="header-logo">
        <div className="logo-icon">⚖️</div>
        <div className="logo-text">
          <h1>AI Devil's Advocate</h1>
          <span>Legal Risk Intelligence</span>
        </div>
      </div>
      <div className="header-badge">
        <span className="badge-dot" />
        InLegalBERT · Fine-tuned on CUAD
      </div>
    </header>
  )
}

function Hero() {
  return (
    <div className="hero">
      <div className="hero-eyebrow">
        ⚡ Powered by your custom-trained Legal AI
      </div>
      <h2>
        Your contract's<br />
        <span className="gradient-text">hidden risks, exposed.</span>
      </h2>
      <p>
        Upload any legal contract as a PDF. Our fine-tuned AI will scan it for
        15 categories of high-risk clauses and give you a full risk report in seconds.
      </p>
    </div>
  )
}

function UploadZone({ onAnalyze, isLoading }) {
  const fileInputRef = useRef()
  const [dragging, setDragging] = useState(false)

  const handleFile = useCallback((file) => {
    if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
      alert('Please upload a PDF file.')
      return
    }
    onAnalyze(file)
  }, [onAnalyze])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }, [handleFile])

  return (
    <div className="upload-section">
      <div
        id="upload-drop-zone"
        className={`upload-zone ${dragging ? 'dragging' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !isLoading && fileInputRef.current?.click()}
      >
        <div className="upload-icon">📄</div>
        <h3>Drop your contract here</h3>
        <p>Supports any legal PDF — NDA, SaaS agreements, employment contracts, and more</p>
        <button
          id="select-file-btn"
          className="upload-btn"
          disabled={isLoading}
          onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}
        >
          📂 Select PDF File
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>
    </div>
  )
}

function LoadingView({ step }) {
  return (
    <div className="loading-overlay">
      <div className="loading-spinner" />
      <p style={{ fontWeight: 600, fontSize: 18, marginBottom: 8 }}>Analyzing your contract…</p>
      <p style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 24 }}>
        The AI is reading every clause. This usually takes 15–30 seconds.
      </p>
      <div className="loading-steps">
        {LOADING_STEPS.map((s, i) => (
          <div
            key={i}
            className={`loading-step ${i < step ? 'done' : i === step ? 'active' : ''}`}
          >
            {i < step ? '✅' : i === step ? '⏳' : '○'} {s}
          </div>
        ))}
      </div>
    </div>
  )
}

function FindingCard({ finding, index }) {
  const isFound = finding.found
  const answerPreview = finding.answer
    ? finding.answer.slice(0, 200) + (finding.answer.length > 200 ? '…' : '')
    : null

  return (
    <div
      className={`finding-card ${isFound ? `found severity-${finding.severity}` : ''}`}
      style={{ animationDelay: `${index * 0.04}s` }}
    >
      <div className="card-header">
        <div className="card-icon-wrap">{finding.icon}</div>
        <div className="card-title">
          <h4>{finding.label}</h4>
          <p>{finding.description}</p>
        </div>
        <span className={`severity-badge ${finding.severity}`}>
          {finding.severity}
        </span>
      </div>

      {isFound && answerPreview && (
        <div className="answer-excerpt">
          <strong>AI Extracted Clause</strong>
          "{answerPreview}"
        </div>
      )}

      {isFound && finding.precedents && finding.precedents.length > 0 && (
        <div className="precedents-section">
          <h5>📚 Relevant Historical Precedents</h5>
          <div className="precedent-list">
            {finding.precedents.map((p, i) => (
              <div key={i} className="precedent-item">
                <div className="precedent-meta">
                  <span className={`severity-badge ${p.risk_tier?.toLowerCase()}`}>{p.risk_tier} Risk</span>
                  <span className="precedent-source">{p.source_doc}</span>
                </div>
                <p>"{p.text.slice(0, 150)}{p.text.length > 150 ? '...' : ''}"</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {isFound && (finding.lawyer_take || finding.optimist_take || finding.pessimist_take) && (
        <div className="persona-grid">
          {finding.lawyer_take && (
            <div className="persona-card lawyer">
              <div className="persona-header">💼 Loophole Lawyer</div>
              <p className="persona-summary">{finding.lawyer_take.summary}</p>
              <ul className="persona-points">
                {finding.lawyer_take.key_points?.slice(0,2).map((k,i)=><li key={i}>{k}</li>)}
              </ul>
            </div>
          )}
          {finding.optimist_take && (
            <div className="persona-card optimist">
              <div className="persona-header">🤝 Deal Optimist</div>
              <p className="persona-summary">{finding.optimist_take.summary}</p>
              <ul className="persona-points">
                {finding.optimist_take.key_points?.slice(0,2).map((k,i)=><li key={i}>{k}</li>)}
              </ul>
            </div>
          )}
          {finding.pessimist_take && (
            <div className="persona-card pessimist">
              <div className="persona-header">🛡️ Risk Analyst</div>
              <p className="persona-summary">{finding.pessimist_take.summary}</p>
              <ul className="persona-points">
                {finding.pessimist_take.key_points?.slice(0,2).map((k,i)=><li key={i}>{k}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}

      <div className="card-status">
        <div className={`status-indicator ${isFound ? 'found' : 'not-found'}`} />
        <span className="status-text">
          {isFound ? '🚨 Clause detected — review carefully' : '✅ Not detected in this contract'}
        </span>
      </div>
    </div>
  )
}

function ResultsDashboard({ result, onReset }) {
  const [filter, setFilter] = useState('all')

  const riskLevel = getRiskLevel(result.risk_score)
  const riskLabel = getRiskLabel(result.risk_score)

  const filteredFindings = result.findings.filter((f) => {
    if (filter === 'all') return true
    if (filter === 'found') return f.found
    if (filter === 'high') return f.severity === 'high'
    if (filter === 'medium') return f.severity === 'medium'
    if (filter === 'low') return f.severity === 'low'
    return true
  })

  return (
    <div className="results-section">
      <button id="analyze-another-btn" className="btn-secondary" onClick={onReset}>
        ← Analyze Another Contract
      </button>

      <div className="results-header">
        <div className="results-title">
          <h2>Risk Analysis Report</h2>
          <p>
            📄 {result.filename} · {result.word_count.toLocaleString()} words ·{' '}
            {result.findings.filter(f => f.found).length} of {result.findings.length} risk categories detected
          </p>
        </div>

        <div className="risk-score-card">
          <div className="score-label">Overall Risk Score</div>
          <div className={`score-number ${riskLevel}`}>{result.risk_score}</div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4 }}>{riskLabel}</div>
          <div className="score-bar">
            <div
              className={`score-bar-fill ${riskLevel}`}
              style={{ width: `${result.risk_score}%` }}
            />
          </div>
        </div>
      </div>

      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-icon high">🔴</div>
          <div className="stat-info">
            <div className="num high">{result.high_risks}</div>
            <div className="lbl">High Severity Risks</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon medium">🟡</div>
          <div className="stat-info">
            <div className="num medium">{result.medium_risks}</div>
            <div className="lbl">Medium Severity Risks</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon low">🟢</div>
          <div className="stat-info">
            <div className="num low">{result.low_risks}</div>
            <div className="lbl">Low Severity Risks</div>
          </div>
        </div>
      </div>

      <div className="filter-tabs">
        {FILTERS.map((f) => (
          <button
            key={f.key}
            id={`filter-${f.key}`}
            className={`filter-tab ${filter === f.key ? 'active' : ''}`}
            onClick={() => setFilter(f.key)}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div className="findings-grid">
        {filteredFindings.map((finding, i) => (
          <FindingCard key={finding.id} finding={finding} index={i} />
        ))}
      </div>
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────
export default function App() {
  const [phase, setPhase] = useState('upload') // 'upload' | 'loading' | 'results' | 'error'
  const [loadingStep, setLoadingStep] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const runLoadingAnimation = () => {
    let step = 0
    const interval = setInterval(() => {
      step++
      setLoadingStep(step)
      if (step >= LOADING_STEPS.length - 1) clearInterval(interval)
    }, 1800)
    return interval
  }

  const handleAnalyze = async (file) => {
    setPhase('loading')
    setLoadingStep(0)
    setError(null)

    const interval = runLoadingAnimation()

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData,
      })

      clearInterval(interval)

      if (!response.ok) {
        // Safely parse error — backend may return JSON or plain text
        let errorMsg = `Server error ${response.status}`
        try {
          const errData = await response.json()
          errorMsg = errData.detail || errData.message || errorMsg
        } catch {
          const text = await response.text().catch(() => '')
          if (text) errorMsg = text.slice(0, 200)
        }
        throw new Error(errorMsg)
      }

      const data = await response.json()
      setResult(data)
      setPhase('results')
    } catch (err) {
      clearInterval(interval)
      setError(err.message || 'Could not connect to the AI server. Make sure the backend is running.')
      setPhase('error')
    }
  }

  const handleReset = () => {
    setPhase('upload')
    setResult(null)
    setError(null)
    setLoadingStep(0)
  }

  return (
    <div className="app">
      <Header />
      {phase === 'upload' && (
        <>
          <Hero />
          <UploadZone onAnalyze={handleAnalyze} isLoading={false} />
        </>
      )}
      {phase === 'loading' && <LoadingView step={loadingStep} />}
      {phase === 'error' && (
        <>
          <div className="error-card">
            <h3>⚠️ Analysis Failed</h3>
            <p>{error}</p>
          </div>
          <UploadZone onAnalyze={handleAnalyze} isLoading={false} />
        </>
      )}
      {phase === 'results' && result && (
        <ResultsDashboard result={result} onReset={handleReset} />
      )}
    </div>
  )
}
