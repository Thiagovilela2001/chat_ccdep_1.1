import { useEffect, useRef } from "react";
import {
  ArrowUp,
  BookOpen,
  CircleStop,
} from "lucide-react";

export function Composer({ value, setValue, onSubmit, loading, onCancel }) {
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
      </form>
      <small className="composer-note">A Nadia pode cometer erros. Confirme informações importantes nas fontes.</small>
    </div>
  );
}
