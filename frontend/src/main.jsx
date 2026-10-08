import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./styles.css";

async function bootstrap() {
  const rootElement = document.getElementById("root");
  if (!rootElement) throw new Error("Elemento #root não encontrado no documento.");

  const [{ default: App }, { default: AppErrorBoundary }] = await Promise.all([
    import("./App"),
    import("./AppErrorBoundary"),
  ]);

  rootElement.dataset.nadiaMounted = "true";
  createRoot(rootElement).render(
    <StrictMode>
      <AppErrorBoundary>
        <App />
      </AppErrorBoundary>
    </StrictMode>,
  );
}

bootstrap().catch((error) => {
  console.error("Falha no bootstrap da interface.", error);
  window.__showNadiaBootstrapError?.(error);
});
