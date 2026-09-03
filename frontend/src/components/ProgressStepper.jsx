import { useEffect, useState } from "react";
import { CheckCircle2, Clock3, Sparkles } from "lucide-react";

const STEPS = [
  { id: "interpret", label: "Interpretação & Expansão da Consulta", estimatedSec: 2 },
  { id: "retrieval", label: "Busca Híbrida (Vetores, BM25 & Séries)", estimatedSec: 6 },
  { id: "synthesis", label: "Síntese Analítica & Cruzamento de Dados", estimatedSec: 14 },
  { id: "validation", label: "Validação Numérica & Citações", estimatedSec: 20 },
];

export function ProgressStepper({ active = true }) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    if (!active) {
      setSeconds(0);
      return;
    }
    const timer = setInterval(() => {
      setSeconds((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, [active]);

  const currentStepIndex = Math.min(
    STEPS.findIndex((s) => seconds < s.estimatedSec),
    STEPS.length - 1
  );
  const activeIndex = currentStepIndex === -1 ? STEPS.length - 1 : currentStepIndex;

  return (
    <div className="progress-stepper-container" role="status" aria-live="polite">
      <div className="progress-stepper-header">
        <div className="progress-stepper-title">
          <Sparkles className="progress-sparkle-icon spin-slow" size={16} />
          <span>Processando análise documental</span>
        </div>
        <div className="progress-stepper-timer">
          <Clock3 size={14} />
          <span>{seconds}s</span>
        </div>
      </div>

      <div className="progress-stepper-steps">
        {STEPS.map((step, idx) => {
          const isDone = idx < activeIndex;
          const isCurrent = idx === activeIndex;

          return (
            <div
              key={step.id}
              className={`progress-step-item ${isDone ? "is-done" : ""} ${isCurrent ? "is-current" : ""}`}
            >
              <div className="progress-step-indicator">
                {isDone ? (
                  <CheckCircle2 size={16} className="step-icon-done" />
                ) : (
                  <div className={`step-dot ${isCurrent ? "step-dot-pulse" : ""}`} />
                )}
              </div>
              <span className="progress-step-label">{step.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default ProgressStepper;
