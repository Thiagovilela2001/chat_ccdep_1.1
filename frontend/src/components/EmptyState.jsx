import {
  CheckCircle2,
  ListChecks,
  ShieldCheck,
} from "lucide-react";

import { BrandMark } from "./BrandMark";
import { Composer } from "./Composer";

export function EmptyState({ input, setInput, onSubmit }) {
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
    </section>
  );
}
