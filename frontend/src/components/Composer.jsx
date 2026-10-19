import { useEffect, useRef } from "react";
import {
  ArrowUp,
  BookOpen,
  CircleStop,
  FileText,
} from "lucide-react";

export function Composer({ value, setValue, onSubmit, loading, onCancel, articleMode = false, setArticleMode }) {
  const textarea = useRef(null);

  useEffect(() => {
    const element = textarea.current;
    if (!element) return;
    element.style.height = "0px";
    element.style.height = `${Math.min(element.scrollHeight, 160)}px`;
  }, [value]);

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!loading && value.trim()) onSubmit(value);
    }
  }

  return (
    <div className="composer-wrap">
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          if (!loading && value.trim()) onSubmit(value);
        }}
      >
        <textarea
          ref={textarea}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Pergunte alguma coisa sobre São Paulo"
          rows="1"
          maxLength="4000"
          aria-label="Digite sua pergunta"
        />
        <div className="composer-bottom">
          <span><BookOpen size={14} /> Resposta com fontes do Seade</span>
          <div className="composer-actions">
            {setArticleMode && (
              <button
                className={`mode-toggle ${articleMode ? "is-on" : ""}`}
                type="button"
                aria-pressed={articleMode}
                disabled={loading}
                onClick={() => setArticleMode(!articleMode)}
                title="Raciocínio mais cuidadoso nas conclusões, para análises e artigos. A resposta leva cerca de 1 minuto."
              >
                <FileText size={14} /> Modo artigo
              </button>
            )}
            {loading ? (
              <button className="send-button is-stop" type="button" onClick={onCancel} aria-label="Cancelar consulta">
                <CircleStop size={18} />
              </button>
            ) : (
              <button className="send-button" type="submit" disabled={!value.trim()} aria-label="Enviar pergunta">
                <ArrowUp size={19} />
              </button>
            )}
          </div>
        </div>
      </form>
      <small className="composer-note">
        {articleMode
          ? "Modo artigo ligado: conclusões mais rigorosas, em cerca de 1 minuto por resposta."
          : "A Nadia pode cometer erros. Confirme informações importantes nas fontes."}
      </small>
    </div>
  );
}
