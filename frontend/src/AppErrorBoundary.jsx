import { Component } from "react";

const NADIA_LOCAL_KEYS = [
  "nadia.messages.v1",
  "nadia.rag.v1",
  "nadia.theme.v1",
  "nadia.endpoints.v1",
];

export default class AppErrorBoundary extends Component {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error, info) {
    console.error("Falha ao inicializar a interface da Nadia.", error, info);
  }

  resetApplication = () => {
    try {
      NADIA_LOCAL_KEYS.forEach((key) => localStorage.removeItem(key));
      sessionStorage.removeItem("nadia.apiKey.v1");
    } catch {
      // O recarregamento ainda pode recuperar a interface sem acesso ao storage.
    }
    window.location.reload();
  };

  render() {
    if (!this.state.failed) return this.props.children;

    return (
      <main
        style={{
          minHeight: "100dvh",
          display: "grid",
          placeItems: "center",
          padding: 24,
          background: "#f5f5f6",
          color: "#282828",
          fontFamily: 'Arial, "Segoe UI", sans-serif',
        }}
      >
        <section style={{ width: "min(520px, 100%)", padding: 28, background: "#fff", border: "1px solid #dedfe2" }}>
          <small style={{ color: "#2868f0", fontWeight: 800 }}>NADIA · SEADE</small>
          <h1 style={{ margin: "12px 0 8px", fontSize: 24 }}>A interface não conseguiu iniciar</h1>
          <p style={{ margin: "0 0 20px", color: "#5e6268", lineHeight: 1.6 }}>
            Uma preferência salva no navegador pode estar incompatível com esta versão.
            Restaure os dados locais para abrir uma sessão limpa.
          </p>
          <button
            type="button"
            onClick={this.resetApplication}
            style={{
              padding: "10px 14px",
              color: "#fff",
              fontWeight: 700,
              cursor: "pointer",
              background: "#2868f0",
              border: 0,
              borderRadius: 4,
            }}
          >
            Restaurar e recarregar
          </button>
        </section>
      </main>
    );
  }
}

