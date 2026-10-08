import { useEffect, useRef, useState } from "react";
import {
  X,
} from "lucide-react";

import { ACTIVE_RAG_TYPE } from "../lib/api";

export function SettingsDialog({ open, onClose, endpoints, setEndpoints, apiKey, setApiKey, developerMode, setDeveloperMode }) {
  const [draft, setDraft] = useState(endpoints[ACTIVE_RAG_TYPE] || "");
  const closeButtonRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const previousFocus = document.activeElement;
    const frame = window.requestAnimationFrame(() => closeButtonRef.current?.focus());
    const handleKeyDown = (event) => {
      if (event.key === "Escape") onClose?.();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener("keydown", handleKeyDown);
      previousFocus?.focus?.();
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="settings-dialog" role="dialog" aria-modal="true" aria-labelledby="settings-title">
        <header>
          <div><small>Preferências</small><h2 id="settings-title">Configurações</h2></div>
          <button ref={closeButtonRef} className="icon-button" onClick={onClose} aria-label="Fechar configurações"><X size={18} /></button>
        </header>
        <div className="dialog-body">
          <label>
            <span>Endpoint do assistente</span>
            <input value={draft} onChange={(event) => setDraft(event.target.value)} spellCheck="false" />
            <small>O endereço é salvo somente neste navegador.</small>
          </label>
          <label>
            <span>Chave da API</span>
            <input type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="Opcional" autoComplete="off" />
            <small>A chave fica apenas na sessão atual e nunca é persistida no histórico.</small>
          </label>
          <label className="toggle-row">
            <span><strong>Modo desenvolvedor</strong><small>Exibe roteamento, tempos e o JSON técnico.</small></span>
            <input type="checkbox" checked={developerMode} onChange={(event) => setDeveloperMode(event.target.checked)} />
            <i />
          </label>
        </div>
        <footer>
          <button className="secondary-button" onClick={onClose}>Cancelar</button>
          <button className="primary-button" onClick={() => { setEndpoints({ ...endpoints, [ACTIVE_RAG_TYPE]: draft.trim().replace(/\/+$/, "") }); onClose(); }}>Salvar alterações</button>
        </footer>
      </section>
    </div>
  );
}
