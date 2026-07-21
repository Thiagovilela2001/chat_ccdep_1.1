````bash
cat << 'EOF' > .claude/skills/shadcn/SKILL.md
---
name: "shadcn"
description: "Instala, configura, utiliza e personaliza componentes do shadcn/ui por meio da CLI oficial. Use esta skill ao criar formulários, diálogos, tabelas, menus, cards, layouts e outros elementos de interface no projeto atual."
version: "1.1.0"
---

# Skill de shadcn/ui

Você é especialista em React, Tailwind CSS, acessibilidade e no ecossistema shadcn/ui.

Sempre que a tarefa envolver criação, alteração ou composição de elementos de interface, avalie se algum componente do shadcn/ui pode ser utilizado. Prefira componentes existentes da biblioteca em vez de recriar soluções equivalentes do zero.

## Objetivos

- Utilizar a CLI oficial do shadcn/ui para adicionar componentes.
- Manter consistência visual e estrutural com o projeto existente.
- Evitar dependências, componentes e abstrações desnecessárias.
- Produzir interfaces responsivas, acessíveis e fáceis de manter.
- Preservar personalizações já existentes no projeto.

## Fluxo de trabalho obrigatório

### 1. Inspecionar o projeto

Antes de instalar ou modificar componentes:

1. Verifique se o projeto utiliza React e Tailwind CSS.
2. Verifique a existência dos seguintes arquivos:
   - `components.json`
   - `package.json`
   - `tsconfig.json` ou `jsconfig.json`
   - arquivo de configuração do Tailwind, quando aplicável
3. Identifique:
   - estrutura de diretórios;
   - aliases de importação;
   - componentes já instalados;
   - convenções visuais existentes;
   - gerenciador de pacotes utilizado pelo projeto.
4. Não altere configurações existentes sem necessidade.

### 2. Inicializar o shadcn/ui

Se o arquivo `components.json` não existir, informe que o shadcn/ui ainda não foi inicializado.

Quando a inicialização for necessária para concluir a tarefa, execute:

```bash
npx shadcn@latest init
````

Antes de confirmar as opções da CLI, analise a estrutura e as configurações atuais do projeto. Não escolha caminhos, aliases, estilos ou variáveis CSS incompatíveis com o código existente.

Após a inicialização, confira se o `components.json` foi criado corretamente.

### 3. Selecionar os componentes

Antes de instalar:

1. Identifique quais componentes do shadcn/ui atendem à solicitação.
2. Instale somente os componentes realmente necessários.
3. Não reinstale componentes que já estejam presentes.
4. Reutilize componentes e padrões existentes sempre que possível.

Exemplos:

```bash
npx shadcn@latest add button
npx shadcn@latest add dialog
npx shadcn@latest add form input label
```

Para verificar opções ou nomes de componentes, utilize a CLI oficial em vez de presumir que um componente existe.

### 4. Instalar pela CLI oficial

Adicione os componentes exclusivamente pela CLI oficial:

```bash
npx shadcn@latest add <componente>
```

Não crie manualmente arquivos básicos em `components/ui/` como substituição à instalação pela CLI.

É permitido modificar os arquivos gerados pela CLI quando a tarefa exigir personalização, desde que:

* a API pública do componente seja preservada sempre que possível;
* alterações existentes não sejam apagadas;
* não sejam removidos comportamentos de acessibilidade;
* a personalização permaneça compatível com o restante do projeto.

Nunca sobrescreva silenciosamente um componente que já contenha alterações personalizadas. Analise as diferenças antes de substituí-lo.

### 5. Implementar a interface

Após a instalação:

1. Importe os componentes pelo alias configurado no projeto, normalmente:

```tsx
import { Button } from "@/components/ui/button"
```

2. Monte a interface utilizando:

   * componentes do shadcn/ui;
   * Tailwind CSS;
   * utilitários e componentes já existentes no projeto.

3. Evite:

   * CSS inline sem necessidade;
   * estilos globais para ajustes locais;
   * duplicação de componentes;
   * abstrações prematuras;
   * dependências adicionais quando o shadcn/ui já oferecer a solução.

4. Preserve:

   * tipagem TypeScript;
   * responsividade;
   * navegação por teclado;
   * foco visível;
   * labels e descrições acessíveis;
   * estados de carregamento, erro, vazio e desabilitado quando aplicáveis.

### 6. Componentes client-side

Em projetos com React Server Components, adicione `"use client"` somente nos arquivos que realmente precisarem de:

* estado;
* efeitos;
* eventos do navegador;
* hooks;
* componentes interativos que dependam do cliente.

Não transforme páginas ou layouts inteiros em Client Components sem necessidade.

### 7. Formulários

Ao criar formulários:

* utilize os componentes de formulário já configurados no projeto;
* reutilize bibliotecas existentes, como React Hook Form e Zod, caso já estejam instaladas;
* apresente mensagens de validação próximas aos respectivos campos;
* associe corretamente `Label`, campo, descrição e mensagem de erro;
* não introduza uma nova biblioteca de formulários sem necessidade.

### 8. Validação final

Depois de implementar:

1. Verifique se todos os imports estão corretos.
2. Confirme se não existem componentes ou dependências não utilizados.
3. Execute os comandos de validação disponíveis no projeto, como:

```bash
npm run lint
npm run typecheck
npm run build
```

Execute somente os scripts que estiverem definidos no `package.json`.

4. Corrija erros diretamente relacionados à implementação.
5. Confirme que a interface funciona em telas menores e maiores.
6. Informe resumidamente:

   * componentes instalados;
   * arquivos principais alterados;
   * validações executadas;
   * eventuais limitações encontradas.

## Regras importantes

* Não recrie manualmente um componente disponível no shadcn/ui.
* Não instale componentes que não serão utilizados.
* Não substitua personalizações existentes sem analisar o impacto.
* Não modifique configurações globais apenas para resolver um detalhe local.
* Não remova recursos de acessibilidade fornecidos pelos componentes.
* Não use caminhos de importação diferentes do padrão configurado no `components.json`.
* Não presuma que o projeto utiliza a configuração padrão do shadcn/ui.
* Não considere a tarefa concluída enquanto houver erros de importação, tipagem ou build causados pelas alterações.
  EOF

```
```
