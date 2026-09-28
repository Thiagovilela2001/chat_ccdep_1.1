FROM node:22-alpine AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .

# O Vite embute as VITE_* no bundle em tempo de build. As portas do rag-principal
# e do orquestrador no host podem não ser 8000/8010 (ver RAG_PRINCIPAL_PORT e
# RAG_ORCHESTRATOR_PORT no .env): sem estes args, o bundle apontaria para as
# portas padrão da máquina. Vazio = mantém o padrão do código.
ARG VITE_PRINCIPAL_URL=""
ARG VITE_META_URL=""
ENV VITE_PRINCIPAL_URL=$VITE_PRINCIPAL_URL
ENV VITE_META_URL=$VITE_META_URL

RUN npm run build

FROM node:22-alpine

ENV NODE_ENV=production \
    PORT=8501

WORKDIR /app
COPY --from=build --chown=node:node /app/dist ./dist
COPY --from=build --chown=node:node /app/server.mjs ./server.mjs

USER node
EXPOSE 8501
CMD ["node", "server.mjs"]
