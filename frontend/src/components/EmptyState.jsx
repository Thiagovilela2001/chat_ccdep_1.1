import {
  ArrowUp,
  Baby,
  BookOpen,
  Briefcase,
  Building2,
  ChartLine,
  CheckCircle2,
  Factory,
  HandHeart,
  HeartPulse,
  Hourglass,
  ListChecks,
  MapPin,
  Route,
  Scale,
  ShieldCheck,
  Users,
} from "lucide-react";

import { BrandMark } from "./BrandMark";
import { Composer } from "./Composer";

const SUGGESTION_ICONS = {
  labor: Briefcase,
  social: HandHeart,
  activity: Factory,
  comparison: Scale,
  demography: Users,
  "demography-aging": Hourglass,
  "demography-fertility": Baby,
  "demography-longevity": HeartPulse,
  "demography-migration": Route,
  "demography-projections": ChartLine,
  investment: Building2,
  regional: MapPin,
};

export function EmptyState({ onPick, input, setInput, onSubmit, suggestions }) {
  const visibleSuggestions = suggestions.slice(0, 6);

  return (
    <section className="empty-state">
      <div className="hero-intro">
        <div className="welcome-mark">
          <BrandMark />
          <span>
            <small>Inteligência pública aplicada</small>
            Nadia · Fundação Seade
          </span>
        </div>
        <h1>Transforme dados de São Paulo em decisões mais claras.</h1>
        <p>
          Investigue trabalho, economia e sociedade em linguagem natural. A Nadia cruza
          documentos oficiais e apresenta cada resposta com fontes rastreáveis.
        </p>
      </div>
      <Composer
        value={input}
        setValue={setInput}
        onSubmit={onSubmit}
        loading={false}
        onCancel={() => {}}
      />
      <div className="trust-row" aria-label="Compromissos da resposta">
        <span><ShieldCheck size={15} /> Fontes oficiais</span>
        <span><CheckCircle2 size={15} /> Números conferidos</span>
        <span><ListChecks size={15} /> Síntese objetiva</span>
      </div>

      <section className="suggestion-section" aria-labelledby="suggestion-title">
        <div className="suggestion-label">
          <span>
            <small>Comece por uma pergunta</small>
            <strong id="suggestion-title">Explore os dados do Estado</strong>
          </span>
          <div className="theme-ribbon" aria-label="Áreas de conhecimento">
            <span><Briefcase size={14} /> Trabalho</span>
            <span><ChartLine size={14} /> Economia</span>
            <span><Users size={14} /> Sociedade</span>
          </div>
        </div>
        <div className="suggestion-carousel" aria-label="Perguntas sugeridas">
          {visibleSuggestions.map(({ id, eyebrow, title }) => {
            const isDemography = id.startsWith("demography");
            const Icon = SUGGESTION_ICONS[id] || BookOpen;
            return (
              <button
                type="button"
                className={`suggestion-card ${isDemography ? "is-demography" : ""}`}
                key={id}
                onClick={() => onPick(title)}
              >
                <span className="suggestion-icon"><Icon size={18} /></span>
                <span className="suggestion-copy">
                  <small>{eyebrow}</small>
                  <strong>{title}</strong>
                </span>
                <ArrowUp size={16} />
              </button>
            );
          })}
        </div>
      </section>
    </section>
  );
}
